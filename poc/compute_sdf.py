#!/usr/bin/env python3
"""
SDF computation using mesh_to_sdf library (more robust than TorchSDF).
Requires: pip install mesh-to-sdf
"""
import argparse
import os
import sys
import numpy as np
import trimesh
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import mesh_to_sdf
except ImportError:
    print("Error: mesh_to_sdf not installed. Install with: pip install mesh-to-sdf")
    sys.exit(1)

import torch
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix, Transform3d, quaternion_invert, quaternion_apply, axis_angle_to_quaternion, quaternion_multiply
import json


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
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = (bbox_max - bbox_min).max()

    # Center and scale to fit within [-target_scale, target_scale]
    normalized_vertices = (mesh.vertices - bbox_center) / (bbox_size / 2 / target_scale)
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


def visualize_mesh(mesh, output_path, verbose=True):
    """Visualize the normalized mesh."""
    if verbose:
        print("Creating mesh visualization...")

    # Sample vertices for visualization
    vertices = mesh.vertices
    max_points = 10000

    if len(vertices) > max_points:
        indices = np.random.choice(len(vertices), max_points, replace=False)
        sampled_vertices = vertices[indices]
        if verbose:
            print(f"  Downsampled to {max_points:,} vertices for visualization")
    else:
        sampled_vertices = vertices

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        sampled_vertices[:, 0],
        sampled_vertices[:, 1],
        sampled_vertices[:, 2],
        c='blue',
        s=1,
        alpha=0.5,
        edgecolors='none'
    )

    # Draw the grid bounds
    from itertools import product
    grid_corners = np.array(list(product([-1, 1], [-1, 1], [-1, 1])))
    ax.scatter(grid_corners[:, 0], grid_corners[:, 1], grid_corners[:, 2],
               c='red', s=100, marker='o', label='Grid bounds [-1,1]³')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Normalized Mesh')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.legend()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"✓ Saved mesh visualization to {output_path}")

    plt.close()


def world_to_local_transform(world_points, target_rotation, target_translation, target_scale, normalization_scale):
    """
    Transform world points to target object's local normalized space [-1, 1]³.

    Args:
        world_points: [N, 3] torch tensor
        target_rotation: [4] quaternion [w, x, y, z]
        target_translation: [3] translation
        target_scale: [3] scale (uniform) - world transform scale
        normalization_scale: float - scale factor from mesh normalization

    Returns:
        [N, 3] points in target's normalized SDF space
    """
    # Use PyTorch3D's Transform3d to get exact inverse of forward transform
    rot_matrix = quaternion_to_matrix(target_rotation.unsqueeze(0)).squeeze()

    # Forward transform: scale -> rotate -> translate
    tfm_forward = Transform3d()
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


def export_pointcloud_ply(target_points, other_points, other_sdf, output_path, verbose=True):
    """
    Export point cloud as PLY file with colors.

    Args:
        target_points: [N, 3] numpy array of target SDF interior points
        other_points: [M, 3] numpy array of other objects' points (or None)
        other_sdf: [M] numpy array of SDF values for other points (or None)
        output_path: path to save PLY file
        verbose: print progress
    """
    if verbose:
        print("Exporting point cloud as PLY...")

    points = []
    colors = []

    # Add target SDF interior points (green/yellow gradient)
    for pt in target_points:
        points.append(pt)
        colors.append([0, 255, 100])  # Green for target interior

    # Add other objects' points (blue=outside, red=inside)
    if other_points is not None and other_sdf is not None:
        for pt, sdf_val in zip(other_points, other_sdf):
            points.append(pt)
            if sdf_val < 0:
                # Penetrating - red
                colors.append([255, 0, 0])
            else:
                # Outside - blue
                colors.append([0, 100, 255])

    points = np.array(points)
    colors = np.array(colors, dtype=np.uint8)

    # Write PLY file
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for pt, col in zip(points, colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {col[0]} {col[1]} {col[2]}\n")

    if verbose:
        print(f"✓ Saved point cloud PLY to {output_path}")
        print(f"  Total points: {len(points):,}")
        print(f"  View online at: https://3dviewer.net/ or https://viewstl.com/")


def visualize_sdf(sdf_grid, resolution, target_obj_name, sam3d_dir, normalization_scale, output_path, max_points=50000, verbose=True):
    """
    Visualize SDF interior voxels with other objects transformed to target's local space.

    Args:
        sdf_grid: [H, W, D] SDF grid for target object
        resolution: grid resolution
        target_obj_name: name of target object (e.g., "object_0")
        sam3d_dir: directory containing GLB files and positions.json
        normalization_scale: scale factor from mesh normalization
        output_path: where to save visualization
        max_points: max points to visualize
        verbose: print progress
    """
    if verbose:
        print("Creating SDF visualization with other objects...")

    # Only show interior voxels (negative distance)
    interior_mask = sdf_grid < 0
    interior_coords = np.argwhere(interior_mask)

    if len(interior_coords) == 0:
        print("  Warning: No interior points found!")
        return

    if verbose:
        print(f"  Interior voxels: {len(interior_coords):,}")

    # Downsample if too many points
    if len(interior_coords) > max_points:
        indices = np.random.choice(len(interior_coords), max_points, replace=False)
        interior_coords = interior_coords[indices]
        interior_values = sdf_grid[interior_mask][indices]
        if verbose:
            print(f"  Downsampled to {max_points:,} points for visualization")
    else:
        interior_values = sdf_grid[interior_mask]

    normalized_points = -1 + (interior_coords / (resolution - 1)) * 2
    color_values = (interior_values - interior_values.min()) / (interior_values.max() - interior_values.min() + 1e-8)

    # Load positions.json
    positions_path = os.path.join(sam3d_dir, "positions.json")
    if not os.path.exists(positions_path):
        if verbose:
            print(f"  Warning: positions.json not found at {positions_path}")
        # Fall back to simple visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            normalized_points[:, 0],
            normalized_points[:, 1],
            normalized_points[:, 2],
            c=color_values,
            cmap='viridis',
            s=5,
            alpha=0.3,
            edgecolors='none'
        )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'SDF Interior Voxels (resolution={resolution}³)')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return

    with open(positions_path, 'r') as f:
        positions = json.load(f)

    # Get target object's transform
    if target_obj_name not in positions:
        if verbose:
            print(f"  Warning: {target_obj_name} not found in positions.json")
        return

    target_transform = positions[target_obj_name]
    target_rotation = torch.tensor(target_transform['rotation'], dtype=torch.float32).flatten()
    target_translation = torch.tensor(target_transform['translation'], dtype=torch.float32).flatten()
    target_scale = torch.tensor(target_transform['scale'], dtype=torch.float32).flatten()

    # Apply -90° rotation around X-axis correction
    # This converts from SAM-3D's internal coordinate system to visualization coordinate system
    correction_quat = axis_angle_to_quaternion(torch.tensor([-np.pi/2, 0.0, 0.0]))
    target_rotation = quaternion_multiply(correction_quat, target_rotation.unsqueeze(0)).squeeze()

    if verbose:
        print(f"  Target object: {target_obj_name}")
        print(f"  Loading other objects from {sam3d_dir}...")

    # Load and transform other objects
    other_object_points = []
    other_object_sdf_values = []

    for obj_name, obj_transform in positions.items():
        if obj_name == target_obj_name:
            continue  # Skip target object

        # Extract object index
        obj_idx = obj_name.split('_')[1]
        glb_path = os.path.join(sam3d_dir, f"object_{obj_idx}.glb")

        if not os.path.exists(glb_path):
            if verbose:
                print(f"    Skipping {obj_name}: GLB not found")
            continue

        # Load mesh
        mesh = trimesh.load(glb_path, force='mesh')

        # Sample points from mesh surface (in original GLB coordinate system)
        num_samples = min(10000, len(mesh.vertices))
        points_local, _ = trimesh.sample.sample_surface(mesh, num_samples)

        # Transform to world space using object's transform
        obj_rotation = torch.tensor(obj_transform['rotation'], dtype=torch.float32).flatten()
        obj_translation = torch.tensor(obj_transform['translation'], dtype=torch.float32).flatten()
        obj_scale = torch.tensor(obj_transform['scale'], dtype=torch.float32).flatten()

        # Apply -90° rotation around X-axis correction
        obj_rotation = quaternion_multiply(correction_quat, obj_rotation.unsqueeze(0)).squeeze()

        rot_matrix = quaternion_to_matrix(obj_rotation.unsqueeze(0)).squeeze()
        tfm = Transform3d()
        tfm = tfm.scale(obj_scale.unsqueeze(0)).rotate(rot_matrix.unsqueeze(0)).translate(obj_translation.unsqueeze(0))

        points_local_torch = torch.tensor(points_local, dtype=torch.float32)
        points_world = tfm.transform_points(points_local_torch)

        # Transform from world to target's normalized SDF space
        points_target_local = world_to_local_transform(
            points_world, target_rotation, target_translation, target_scale, normalization_scale
        )

        # Filter points within [-1, 1]³ bounds
        in_bounds = (
            (points_target_local[:, 0] >= -1) & (points_target_local[:, 0] <= 1) &
            (points_target_local[:, 1] >= -1) & (points_target_local[:, 1] <= 1) &
            (points_target_local[:, 2] >= -1) & (points_target_local[:, 2] <= 1)
        )

        points_in_bounds = points_target_local[in_bounds]

        if len(points_in_bounds) == 0:
            if verbose:
                print(f"    {obj_name}: No points within SDF bounds")
            continue

        if verbose:
            print(f"    {obj_name}: {len(points_in_bounds):,} / {len(points_local):,} points within bounds")

        # Query SDF at these points
        sdf_values = query_sdf_trilinear(sdf_grid, points_in_bounds)

        other_object_points.append(points_in_bounds.cpu().numpy())
        other_object_sdf_values.append(sdf_values.cpu().numpy())

    # Create visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot SDF interior voxels (target object)
    scatter1 = ax.scatter(
        normalized_points[:, 0],
        normalized_points[:, 1],
        normalized_points[:, 2],
        c=color_values,
        cmap='viridis',
        s=5,
        alpha=0.3,
        edgecolors='none',
        label='Target SDF interior'
    )

    # Plot other objects' points colored by SDF value
    if other_object_points:
        all_other_points = np.vstack(other_object_points)
        all_other_sdf = np.concatenate(other_object_sdf_values)

        # Downsample if too many
        if len(all_other_points) > max_points:
            indices = np.random.choice(len(all_other_points), max_points, replace=False)
            all_other_points = all_other_points[indices]
            all_other_sdf = all_other_sdf[indices]
            if verbose:
                print(f"  Downsampled other objects to {max_points:,} points")

        # Separate positive and negative distances
        positive_mask = all_other_sdf >= 0
        negative_mask = all_other_sdf < 0

        # Plot positive distances (outside) in blue
        if positive_mask.sum() > 0:
            ax.scatter(
                all_other_points[positive_mask, 0],
                all_other_points[positive_mask, 1],
                all_other_points[positive_mask, 2],
                c='blue',
                s=5,
                alpha=0.5,
                edgecolors='none',
                label=f'Other objects (outside, SDF≥0): {positive_mask.sum():,} pts'
            )

        # Plot negative distances (penetrating) in red
        if negative_mask.sum() > 0:
            ax.scatter(
                all_other_points[negative_mask, 0],
                all_other_points[negative_mask, 1],
                all_other_points[negative_mask, 2],
                c='red',
                s=5,
                alpha=0.8,
                edgecolors='none',
                label=f'Other objects (penetrating, SDF<0): {negative_mask.sum():,} pts'
            )

        if verbose:
            print(f"  Other objects - Positive: {positive_mask.sum():,}, Negative: {negative_mask.sum():,}")

    # Create a figure with 8 subplots showing different angles
    fig_multi = plt.figure(figsize=(24, 16))

    # Define 8 different viewing angles varying elev and roll (azim fixed at 90)
    views = [
        {'elev': 0, 'azim': 90, 'roll': 0, 'name': 'View 1 (elev=0, azim=90, roll=0)'},
        {'elev': 0, 'azim': 90, 'roll': 90, 'name': 'View 2 (elev=0, azim=90, roll=90)'},
        {'elev': 30, 'azim': 90, 'roll': 0, 'name': 'View 3 (elev=30, azim=90, roll=0)'},
        {'elev': 30, 'azim': 90, 'roll': 90, 'name': 'View 4 (elev=30, azim=90, roll=90)'},
        {'elev': 60, 'azim': 90, 'roll': 0, 'name': 'View 5 (elev=60, azim=90, roll=0)'},
        {'elev': 60, 'azim': 90, 'roll': 90, 'name': 'View 6 (elev=60, azim=90, roll=90)'},
        {'elev': 90, 'azim': 90, 'roll': 0, 'name': 'View 7 (elev=90, azim=90, roll=0)'},
        {'elev': 90, 'azim': 90, 'roll': 90, 'name': 'View 8 (elev=90, azim=90, roll=90)'},
    ]

    for idx, view in enumerate(views):
        ax = fig_multi.add_subplot(2, 4, idx + 1, projection='3d')

        # Plot SDF interior voxels (target object)
        scatter1 = ax.scatter(
            normalized_points[:, 0],
            normalized_points[:, 1],
            normalized_points[:, 2],
            c=color_values,
            cmap='viridis',
            s=5,
            alpha=0.3,
            edgecolors='none',
            label='Target SDF interior'
        )

        # Plot other objects' points colored by SDF value
        if other_object_points:
            all_other_points_view = np.vstack(other_object_points)
            all_other_sdf_view = np.concatenate(other_object_sdf_values)

            # Downsample if too many
            if len(all_other_points_view) > max_points:
                indices = np.random.choice(len(all_other_points_view), max_points, replace=False)
                all_other_points_view = all_other_points_view[indices]
                all_other_sdf_view = all_other_sdf_view[indices]

            # Separate positive and negative distances
            positive_mask_view = all_other_sdf_view >= 0
            negative_mask_view = all_other_sdf_view < 0

            # Plot positive distances (outside) in blue
            if positive_mask_view.sum() > 0:
                ax.scatter(
                    all_other_points_view[positive_mask_view, 0],
                    all_other_points_view[positive_mask_view, 1],
                    all_other_points_view[positive_mask_view, 2],
                    c='blue',
                    s=5,
                    alpha=0.5,
                    edgecolors='none',
                    label=f'Outside (SDF≥0): {positive_mask_view.sum():,}'
                )

            # Plot negative distances (penetrating) in red
            if negative_mask_view.sum() > 0:
                ax.scatter(
                    all_other_points_view[negative_mask_view, 0],
                    all_other_points_view[negative_mask_view, 1],
                    all_other_points_view[negative_mask_view, 2],
                    c='red',
                    s=5,
                    alpha=0.8,
                    edgecolors='none',
                    label=f'Penetrating (SDF<0): {negative_mask_view.sum():,}'
                )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(view['name'])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.legend(loc='upper right', fontsize=6)

        # Set view angle
        ax.view_init(elev=view['elev'], azim=view['azim'], roll=view['roll'])

    plt.suptitle(f'SDF Visualization - Target: {target_obj_name} (resolution={resolution}³)', fontsize=16)
    plt.tight_layout()

    # Save multi-view version
    multi_view_path = str(output_path).replace('_viz.png', '_viz_multiview.png')
    plt.savefig(multi_view_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"✓ Saved multi-view visualization to {multi_view_path}")

    plt.close()

    # Export point cloud as PLY for external viewing
    ply_path = str(output_path).replace('_viz.png', '_pointcloud.ply')
    export_pointcloud_ply(
        normalized_points,
        all_other_points if other_object_points else None,
        all_other_sdf if other_object_points else None,
        ply_path,
        verbose
    )


def save_sdf(sdf_grid, normalization_scale, output_path, resolution, mesh=None):
    """Save SDF data to pickle file."""
    sdf_data = {
        'grid': sdf_grid,
        'normalization_scale': normalization_scale,
        'resolution': resolution,
        'mesh': mesh
    }

    with open(output_path, 'wb') as f:
        pickle.dump(sdf_data, f)

    print(f"✓ Saved SDF data to {output_path}")
    print(f"  Grid shape: {sdf_grid.shape}")
    print(f"  Normalization scale: {normalization_scale:.6f}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Compute SDF grid using mesh_to_sdf")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input GLB file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output pickle file (default: input_sdf.pkl)")
    parser.add_argument("--resolution", type=int, default=128,
                        help="Grid resolution NxNxN (default: 128)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Target normalization scale - mesh fits within [-scale, scale]³ (default: 1.0)")
    parser.add_argument("--save-mesh", action="store_true",
                        help="Include mesh in pickle for analytic fallback")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization generation")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    verbose = not args.quiet

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Set output paths
    if args.output is None:
        input_path = Path(args.input)
        args.output = input_path.parent / f"{input_path.stem}_sdf.pkl"

    viz_path = Path(args.output).parent / f"{Path(args.output).stem}_viz.png"
    mesh_viz_path = Path(args.output).parent / f"{Path(args.output).stem}_mesh.png"

    # Load mesh
    if verbose:
        print(f"Loading mesh from {args.input}...")
    mesh = trimesh.load(args.input, force='mesh')

    if verbose:
        print(f"  Vertices: {len(mesh.vertices):,}")
        print(f"  Faces: {len(mesh.faces):,}")
        print(f"  Bounds: min={mesh.bounds[0]}, max={mesh.bounds[1]}")

    # Normalize mesh
    if verbose:
        print(f"\nNormalizing mesh to [-{args.scale}, {args.scale}]³...")
    normalized_mesh, normalization_scale = normalize_mesh(mesh, target_scale=args.scale)

    if verbose:
        print(f"  Normalization scale: {normalization_scale:.6f}")
        print(f"  Normalized bounds: min={normalized_mesh.bounds[0]}, max={normalized_mesh.bounds[1]}")
        print(f"  Normalized vertices min: {normalized_mesh.vertices.min(axis=0)}")
        print(f"  Normalized vertices max: {normalized_mesh.vertices.max(axis=0)}")

    # Compute SDF grid
    print()
    sdf_grid = compute_sdf_grid_mesh_to_sdf(
        normalized_mesh,
        resolution=args.resolution,
        verbose=verbose
    )

    # Save
    print()
    save_sdf(
        sdf_grid,
        normalization_scale,
        args.output,
        args.resolution,
        mesh=mesh if args.save_mesh else None
    )

    # Visualize
    if not args.no_viz:
        print()
        # Extract target object name from input path
        input_path = Path(args.input)
        target_obj_name = input_path.stem  # e.g., "object_0"
        sam3d_dir = str(input_path.parent)  # directory containing GLBs and positions.json

        visualize_sdf(
            sdf_grid,
            args.resolution,
            target_obj_name,
            sam3d_dir,
            normalization_scale,
            viz_path,
            verbose=verbose
        )

    return 0


if __name__ == "__main__":
    exit(main())
