#!/usr/bin/env python3
"""
SDF-based pose optimization using gradient descent.
Optimizes object poses to minimize penetration using pre-computed SDF grids.
"""
import argparse
import os
import sys
import numpy as np
import trimesh
import pickle
from pathlib import Path
import json

try:
    import mesh_to_sdf
except ImportError:
    print("Error: mesh_to_sdf not installed. Install with: pip install mesh-to-sdf")
    sys.exit(1)

import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    Transform3d,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_quaternion,
    quaternion_multiply
)

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

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


def compute_sdf_grid_mesh_to_sdf(mesh, resolution=128, verbose=True):
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


def local_to_world_transform(local_points, rotation_6d, translation, scale):
    """
    Transform points from object's local space to world space.

    Args:
        local_points: [N, 3] torch tensor in object's local space
        rotation_6d: [6] 6D rotation representation
        translation: [3] translation
        scale: [3] scale

    Returns:
        [N, 3] points in world space
    """
    # Convert 6D rotation to rotation matrix
    rot_matrix = rotation_6d_to_matrix(rotation_6d.unsqueeze(0)).squeeze()

    # Forward transform: scale -> rotate -> translate
    tfm_forward = Transform3d(device=local_points.device)
    tfm_forward = tfm_forward.scale(scale.unsqueeze(0)).rotate(rot_matrix.unsqueeze(0)).translate(translation.unsqueeze(0))

    # Apply transformation
    world_points = tfm_forward.transform_points(local_points)

    return world_points


def world_to_local_transform(world_points, target_rotation_6d, target_translation, target_scale, normalization_scale):
    """
    Transform world points to target object's local normalized space [-1, 1]³.

    Args:
        world_points: [N, 3] torch tensor
        target_rotation_6d: [6] 6D rotation representation
        target_translation: [3] translation
        target_scale: [3] scale (uniform) - world transform scale
        normalization_scale: float - scale factor from mesh normalization

    Returns:
        [N, 3] points in target's normalized SDF space
    """
    # Convert 6D rotation to rotation matrix
    rot_matrix = rotation_6d_to_matrix(target_rotation_6d.unsqueeze(0)).squeeze()

    # Forward transform: scale -> rotate -> translate
    tfm_forward = Transform3d(device=world_points.device)
    tfm_forward = tfm_forward.scale(target_scale.unsqueeze(0)).rotate(rot_matrix.unsqueeze(0)).translate(target_translation.unsqueeze(0))

    # Get inverse transform (world -> target's local GLB space)
    tfm_inverse = tfm_forward.inverse()
    points_target_local = tfm_inverse.transform_points(world_points)

    # Apply normalization to get into SDF grid space
    # The SDF was computed on a mesh normalized by dividing by normalization_scale
    points_normalized = points_target_local / normalization_scale

    return points_normalized


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


def optimize_sdf(sdf_grids, transformations, sampled_points, num_iterations=100, learning_rate=0.001):
    """
    Optimize object poses to minimize SDF penetration using gradient descent.

    Args:
        sdf_grids: dict of {name: {"grid": [N,N,N] torch tensor, "scale": float}}
        transformations: dict of {name: {"rotation": [4] torch tensor, "translation": [3] torch tensor, "scale": [3] torch tensor}}
        sampled_points: dict of {name: [10000, 3] torch tensor} sampled points on mesh surface in local space
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

    # Setup optimizer
    optimizers = {}
    for name, params in optim_params.items():
        optimizers[name] = optim.Adam([params['rotation_6d'], params['translation']], lr=learning_rate)

    # Move all data to device before optimization loop
    for name in sdf_grids.keys():
        sdf_grids[name]['grid'] = sdf_grids[name]['grid'].to(device)
    for name in sampled_points.keys():
        sampled_points[name] = sampled_points[name].to(device)
    for name in transformations.keys():
        transformations[name]['scale'] = transformations[name]['scale'].to(device)

    # Optimization loop
    for iteration in range(num_iterations):

        total_loss = 0.0

        # Zero gradients
        for optimizer in optimizers.values():
            optimizer.zero_grad()

        # For each target object, compute penetration loss from other objects
        for name_target in sdf_grids.keys():

            for name in sampled_points.keys():

                if name == name_target:
                    continue  # Skip self

                # 1. Transform points from object's local space to world space
                points_world = local_to_world_transform(
                    sampled_points[name],
                    optim_params[name]['rotation_6d'],
                    optim_params[name]['translation'],
                    transformations[name]['scale']
                )

                # 2. Transform from world to target's local normalized space
                local_points = world_to_local_transform(
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

                # Compute loss: penalize negative SDF values (penetration)
                penetration_loss = F.relu(-sdf_values).mean()

                total_loss += penetration_loss

        # Gradient Descent + Backpropagation
        total_loss.backward()

        for optimizer in optimizers.values():
            optimizer.step()

        print(f"Iteration {iteration+1}/{num_iterations}, Loss: {total_loss.item():.6f}")

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


def main():
    
    # Parser setup
    parser = argparse.ArgumentParser(description="Compute SDF grid using mesh_to_sdf")

    parser.add_argument("--dir", type=str, required=True, default=os.path.join(project_root, "output/sam3d_results"),
                        help="Directory containing input/output files")
    parser.add_argument("--resolution", type=int, default=64,
                        help="SDF resolution NxNxN (default: 64)")
    parser.add_argument("--target-scale", default=0.9, help="Print verbose output")

    args = parser.parse_args()

    print("\nLOADING MESHES\n")

    # Load meshes using trimesh
    glb_files = sorted(Path(args.dir).glob("*.glb"))
    meshes = {}

    print(f"Found {len(glb_files)} .glb files")
    
    for glb_file in glb_files:
        
        print("Loading mesh:", glb_file)

        mesh = trimesh.load(glb_file, force='mesh')
        mesh.merge_vertices()
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.fix_normals()

        meshes[glb_file.stem] = mesh

    print("\nLOADING TRANFORMATIONS\n")

    # Load transformations for each object (local to world)
    # Populate transformations dict with rotation (quaternion), translation, scale tensors
    transformations_file = Path(args.dir) / "positions.json"
    transformations_raw = {}
    transformations = {}

    with open(transformations_file, 'r') as f:
        transformations_raw = json.load(f)

    for name, tfm in transformations_raw.items():

        target_rotation = torch.tensor(tfm['rotation'], dtype=torch.float32).flatten()
        target_translation = torch.tensor(tfm['translation'], dtype=torch.float32).flatten()
        target_scale = torch.tensor(tfm['scale'], dtype=torch.float32).flatten()

        # Apply -90° rotation around X-axis correction
        # This converts from SAM-3D's internal coordinate system to visualization coordinate system
        correction_quat = axis_angle_to_quaternion(torch.tensor([-np.pi/2, 0.0, 0.0]))
        target_rotation = quaternion_multiply(correction_quat, target_rotation.unsqueeze(0)).squeeze()

        transformations[name] = {
            "rotation": target_rotation,
            "translation": target_translation,
            "scale": target_scale
        }

    print("\nSAMPLING MESH POINTS\n")

    # Sample points on mesh surfaces (in local space) for sdf-based optimization
    # These points are in the local mesh space
    sampled_points = {}

    for name, mesh in meshes.items():
        num_samples = min(10000, len(mesh.vertices))
        points_local, _ = trimesh.sample.sample_surface(mesh, num_samples)

        points_tensor = torch.tensor(points_local, dtype=torch.float32)

        # Pad to 10000 points with points far away
        if points_tensor.shape[0] < 10000:
            padding_size = 10000 - points_tensor.shape[0]
            padding_points = torch.full((padding_size, 3), 1e6, dtype=torch.float32)
            points_tensor = torch.cat([points_tensor, padding_points], dim=0)

        sampled_points[name] = points_tensor

    print("\nCOMPUTING SDF\n")

    # Compute SDF grids for each mesh (in normal space)
    # Populate sdf_grids dict with [N, N, N] torch tensors representing SDF values
    sdf_grids = {}

    for name, mesh in meshes.items():
        print(f"Computing mesh: {name}")

        normalized_mesh, normalization_scale = normalize_mesh(mesh, target_scale=args.target_scale)

        sdf_grid = compute_sdf_grid_mesh_to_sdf(
            normalized_mesh,
            resolution=args.resolution,
        )

        sdf_grids[name] = {
            "grid" : torch.tensor(sdf_grid, dtype=torch.float32),
            "scale" : normalization_scale
        }

    # Run optimization
    print("\nOPTIMIZING POSES\n")
    optimized_transforms = optimize_sdf(sdf_grids, transformations, sampled_points)

    # Save optimized transforms to JSON
    output_transforms = {}
    for name, transform in optimized_transforms.items():
        output_transforms[name] = {
            "rotation": transform["rotation"].tolist(),
            "translation": transform["translation"].tolist(),
            "scale": transformations[name]["scale"].cpu().tolist()  # Keep original scale
        }

    output_file = Path(args.dir) / "optimized_positions.json"
    with open(output_file, 'w') as f:
        json.dump(output_transforms, f, indent=2)

    print(f"\nSaved optimized transforms to {output_file}")


if __name__ == "__main__":
    main()
