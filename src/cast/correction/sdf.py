import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_quaternion,
    quaternion_multiply,
)

try:
    import mesh_to_sdf
except ImportError:
    print("Error: mesh_to_sdf not installed. Install with: pip install mesh-to-sdf")
    sys.exit(1)

from cast.correction.transforms import local_to_world_6d, world_to_local_6d


def normalize_mesh(mesh, target_scale=0.8):
    """Normalize mesh to fit within [-target_scale, target_scale]³ while preserving aspect ratio."""
    # Fix mesh before normalization
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.fix_normals()  # Ensure consistent face winding

    # Make sure mesh is watertight if possible
    if not mesh.is_watertight:
        print(f"  Warning: Mesh is not watertight, SDF signs may be unreliable")

    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    bbox_size = (bbox_max - bbox_min).max()

    # Scale to fit within [-target_scale, target_scale] (no centering since SAM-3D already centers)
    normalized_vertices = mesh.vertices / (bbox_size / 2 / target_scale)
    normalized_mesh = mesh.copy()
    normalized_mesh.vertices = normalized_vertices
    normalization_scale = bbox_size / 2 / target_scale

    return normalized_mesh, normalization_scale


def compute_sdf_grid(mesh, resolution=128, verbose=True):
    """
    Compute SDF using mesh_to_sdf library (uses robust winding number approach).

    Args:
        mesh: trimesh.Trimesh (normalized to fit within [-1, 1]³)
        resolution: grid resolution NxNxN
        verbose: print progress
    """
    if verbose:
        print(f"Computing {resolution}³ SDF grid using mesh_to_sdf...")

    # Create grid points
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)
    grid_points = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    query_points = grid_points.reshape(-1, 3)

    if verbose:
        print(f"  Grid points: {len(query_points):,}")

    # Compute SDF at query points
    sdf_values = mesh_to_sdf.mesh_to_sdf(
        mesh,
        query_points,
        surface_point_method='scan',
        sign_method='depth',  # More robust than 'normal'
        scan_count=100,
        scan_resolution=400,
        sample_point_count=10000000,
        normal_sample_count=11
    )

    # Reshape to grid
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)

    if verbose:
        print(f"  SDF range: [{sdf_grid.min():.4f}, {sdf_grid.max():.4f}]")
        inside_points = (sdf_grid < 0).sum()
        print(f"  Inside points: {inside_points:,} ({100*inside_points/sdf_grid.size:.1f}%)")

    return sdf_grid


def query_sdf_trilinear(sdf_grid, local_points):
    """
    Query SDF grid using PyTorch trilinear interpolation.

    Args:
        sdf_grid: [X, Y, Z] numpy array (from meshgrid with indexing='ij')
        local_points: [N, 3] torch tensor in [-1, 1]³ range, coordinates as [x, y, z]

    Returns:
        [N] torch tensor of SDF values
    """
    # Convert to torch if needed
    if isinstance(sdf_grid, np.ndarray):
        sdf_grid = torch.tensor(sdf_grid, dtype=torch.float32)

    # grid_sample expects input as [N, C, D, H, W] where coordinates are [x, y, z] mapping to [W, H, D]
    # Our grid is [X, Y, Z], so we need to permute to [Z, Y, X] to match [D, H, W] = [Z, Y, X]
    # Then coordinates [x, y, z] will correctly index into [X, Y, Z] via the [W, H, D] = [X, Y, Z] mapping
    grid = sdf_grid.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)  # [X,Y,Z] -> [Z,Y,X] -> [1,1,Z,Y,X]

    # Prepare coordinates for grid_sample: [1, 1, 1, N, 3]
    # grid_sample expects coordinates as [..., 3] where the last dimension is [x, y, z]
    # which maps to [W, H, D] = [X, Y, Z] in our permuted grid
    grid_coords = local_points.view(1, 1, 1, -1, 3)

    # Trilinear interpolation
    sampled = F.grid_sample(
        grid, grid_coords,
        mode='bilinear',  # trilinear for 3D
        padding_mode='border',  # clamp to edge
        align_corners=True
    )

    # Extract values: [N]
    return sampled.squeeze()


def _compute_loss(sdf_grids, sampled_points, transformations, optim_params, relations_map, support_sets, device):
    """Compute total loss for a given set of parameters (no gradients needed)."""
    total_loss = torch.tensor(0.0, device=device)

    for name_target in sdf_grids.keys():
        for name in sampled_points.keys():
            if name == name_target:
                continue

            contact_info = relations_map.get(name_target, {}).get(name, None)

            points_world = local_to_world_6d(
                sampled_points[name],
                optim_params[name]['rotation_6d'],
                optim_params[name]['translation'],
                transformations[name]['scale']
            )

            local_points = world_to_local_6d(
                points_world,
                optim_params[name_target]['rotation_6d'],
                optim_params[name_target]['translation'],
                transformations[name_target]['scale'],
                sdf_grids[name_target]['scale']
            )

            sdf_values = query_sdf_trilinear(
                sdf_grids[name_target]['grid'],
                local_points
            )

            name_supports_target = name_target in support_sets.get(name, set())

            if not name_supports_target:
                penetration_loss = F.relu(-sdf_values).mean()
                total_loss += penetration_loss

            if contact_info is not None:
                min_distance = sdf_values.min()
                contact_loss = torch.max(torch.tensor(0.0, device=device), min_distance) * 0.01
                total_loss += contact_loss

                if contact_info.get("contact", "") == "flat":
                    contact_region = (sdf_values > 0) & (sdf_values < 0.1)
                    if contact_region.any():
                        total_loss += sdf_values[contact_region].mean() * 0.1

    return total_loss


def optimize_sdf(sdf_grids, transformations, sampled_points, support_relations, num_iterations=500, learning_rate=0.005, num_restarts=200):
    """
    Optimize object poses to minimize SDF penetration using gradient descent.

    Args:
        sdf_grids: dict of {name: {"grid": [N,N,N] torch tensor, "scale": float}}
        transformations: dict of {name: {"rotation": [4] torch tensor, "translation": [3] torch tensor, "scale": [3] torch tensor}}
        sampled_points: dict of {name: [10000, 3] torch tensor} sampled points on mesh surface in local space
        support_relations: dict with "supports" key mapping supporting object to list of supported objects
        num_iterations: number of optimization iterations
        learning_rate: optimizer learning rate
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare optimizable parameters
    optim_params = {}

    for name, tfm in transformations.items():
        rotation_6d = matrix_to_rotation_6d(
            quaternion_to_matrix(tfm['rotation'].unsqueeze(0)).squeeze()
        )
        translation = tfm['translation']

        optim_params[name] = {
            "rotation_6d": rotation_6d.clone().detach().to(device).requires_grad_(True),
            "translation": translation.clone().detach().to(device).requires_grad_(True)
        }

    # Setup optimizer with momentum and cosine annealing
    optimizers = {}
    schedulers = {}
    for name, params in optim_params.items():
        opt = optim.SGD([params['rotation_6d'], params['translation']],
                        lr=learning_rate, momentum=0.9)
        optimizers[name] = opt
        schedulers[name] = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_iterations, eta_min=0.1 * learning_rate)

    # Move all data to device and normalize SDF grids
    for name in sdf_grids.keys():
        sdf_grids[name]['grid'] = sdf_grids[name]['grid'].to(device)

        # Normalize negative SDF values (interior) so penetration signal is uniform across objects
        grid = sdf_grids[name]['grid']
        neg_mask = grid < 0
        if neg_mask.any():
            neg_max = grid[neg_mask].abs().max()
            print(f"Name {name}: Normalizing SDF grid, neg max = {neg_max:.4f}")
            grid[neg_mask] = grid[neg_mask] / neg_max
        sdf_grids[name]['grid'] = grid
    for name in sampled_points.keys():
        sampled_points[name] = sampled_points[name].to(device)
    for name in transformations.keys():
        transformations[name]['scale'] = transformations[name]['scale'].to(device)

    # Build relations lookup
    # support_relations["relations"] = {object_A: {object_B: {"contact": "flat/point", "type": "support"}}}
    relations_map = support_relations.get("relations", {})

    # Build transitive support sets: for each object, the full set of objects it
    # (transitively) supports via support edges.
    def _build_support_set(source, relations_map):
        """BFS/DFS to find all objects transitively supported by source."""
        visited = set()
        stack = [source]
        while stack:
            node = stack.pop()
            for neighbor, info in relations_map.get(node, {}).items():
                if info.get("type") == "support" and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return visited

    support_sets = {name: _build_support_set(name, relations_map) for name in sdf_grids.keys()}

    for name, supported in support_sets.items():
        if supported:
            print(f"  {name} transitively supports: {sorted(supported)}")
        else:
            print(f"  {name} supports nothing")

    # Random restart search: try N random perturbations of translations (90%-110%)
    # and pick the one with lowest initial loss
    if num_restarts > 1:
        print(f"\nSearching {num_restarts} random initial starts...")
        best_loss = float('inf')
        best_translations = None

        for restart in range(num_restarts):
            trial_params = {}
            for name, params in optim_params.items():
                scale_factor = 0.95 + 0.1 * torch.rand(3, device=device)  # uniform [0.95, 1.05]
                trial_params[name] = {
                    "rotation_6d": params['rotation_6d'].detach(),
                    "translation": params['translation'].detach() * scale_factor
                }

            with torch.no_grad():
                loss = _compute_loss(sdf_grids, sampled_points, transformations, trial_params, relations_map, support_sets, device)

            print(f"  Restart {restart+1}/{num_restarts}: loss = {loss.item():.6f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_translations = {name: trial_params[name]['translation'].clone() for name in trial_params}

        print(f"  Best initial loss: {best_loss:.6f}")
        for name in optim_params:
            optim_params[name]['translation'] = best_translations[name].requires_grad_(True)

    # Build optimizers
    optimizers = {}
    schedulers = {}
    for name, params in optim_params.items():
        opt = optim.SGD([params['rotation_6d'], params['translation']],
                        lr=learning_rate, momentum=0.9)
        optimizers[name] = opt
        schedulers[name] = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_iterations, eta_min=0.1 * learning_rate)

    # Optimization loop
    for iteration in range(num_iterations):

            total_loss = torch.tensor(0.0, device=device)

            # Zero gradients
            for optimizer in optimizers.values():
                optimizer.zero_grad()

            # Loop through all object pairs
            for name_target in sdf_grids.keys():
                for name in sampled_points.keys():

                    if name == name_target:
                        continue  # Skip self

                    contact_info = relations_map.get(name_target, {}).get(name, None)

                    # 1. Transform points from object's local space to world space
                    points_world = local_to_world_6d(
                        sampled_points[name],
                        optim_params[name]['rotation_6d'],
                        optim_params[name]['translation'],
                        transformations[name]['scale']
                    )

                    # Check if target transitively supports source — if so, detach target params
                    # so the supporter doesn't get pushed by this pair's gradients
                    target_supports_source = name in support_sets.get(name_target, set())

                    # 2. Transform from world to target's local normalized space
                    if target_supports_source:
                        local_points = world_to_local_6d(
                            points_world,
                            optim_params[name_target]['rotation_6d'].detach(),
                            optim_params[name_target]['translation'].detach(),
                            transformations[name_target]['scale'],
                            sdf_grids[name_target]['scale']
                        )
                    else:
                        local_points = world_to_local_6d(
                            points_world,
                            optim_params[name_target]['rotation_6d'],
                            optim_params[name_target]['translation'],
                            transformations[name_target]['scale'],
                            sdf_grids[name_target]['scale']
                        )

                    # Query SDF values
                    sdf_values = query_sdf_trilinear(
                        sdf_grids[name_target]['grid'],
                        local_points
                    )

                    # Check if source transitively supports target (if so, skip penetration loss)
                    name_supports_target = name_target in support_sets.get(name, set())

                    if not name_supports_target:
                        # Compute loss: penalize negative SDF values (penetration)
                        penetration_loss = F.relu(-sdf_values).mean()
                        total_loss += penetration_loss

                    if (contact_info is not None):

                        # Penalize minimum distance to prevent objects from drifting apart
                        min_distance = sdf_values.min()
                        contact_loss = torch.max(torch.tensor(0.0, device=device), min_distance) * 0.01
                        total_loss += contact_loss

                        if (contact_info.get("contact", "") == "flat"):

                            # Regularize near-contact region to encourage objects to sit on surfaces
                            contact_region = (sdf_values > 0) & (sdf_values < 0.1)
                            if contact_region.any():
                                regularization_loss = sdf_values[contact_region].mean() * 0.1
                            else:
                                regularization_loss = 0.0

                            total_loss += regularization_loss

            # Gradient Descent + Backpropagation
            total_loss.backward()

            for optimizer in optimizers.values():
                optimizer.step()
            for scheduler in schedulers.values():
                scheduler.step()

            print(f"  Iteration {iteration+1}/{num_iterations}, Loss: {total_loss.item():.6f}")

    # Convert optimized 6D rotations back to quaternions
    optimized_transforms = {}

    # Inverse correction: negate the axis-angle to get inverse of the -90° X-axis rotation
    # This converts back from visualization coordinate system to SAM-3D's coordinate system
    inverse_correction_quat = axis_angle_to_quaternion(torch.tensor([np.pi/2, 0.0, 0.0])).to(device)

    for name, params in optim_params.items():
        # Convert 6D rotation to rotation matrix, then to quaternion
        rot_matrix = rotation_6d_to_matrix(params['rotation_6d'].unsqueeze(0)).squeeze()
        quaternion_corrected = matrix_to_quaternion(rot_matrix.unsqueeze(0)).squeeze()

        # Un-correct: apply inverse correction to get back to original coordinate system
        # This is necessary because visualize_scene.py will apply the correction again
        quaternion_original = quaternion_multiply(inverse_correction_quat, quaternion_corrected.unsqueeze(0)).squeeze()

        optimized_transforms[name] = {
            "rotation": quaternion_original.detach().cpu(),
            "translation": params['translation'].detach().cpu()
        }

    return optimized_transforms
