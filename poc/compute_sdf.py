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


def visualize_sdf(sdf_grid, resolution, mesh, output_path, max_points=50000, verbose=True):
    """Visualize SDF interior voxels alongside the normalized mesh."""
    if verbose:
        print("Creating SDF visualization...")

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

    # Sample mesh vertices
    mesh_vertices = mesh.vertices
    if len(mesh_vertices) > max_points:
        mesh_indices = np.random.choice(len(mesh_vertices), max_points, replace=False)
        sampled_mesh = mesh_vertices[mesh_indices]
    else:
        sampled_mesh = mesh_vertices

    fig = plt.figure(figsize=(20, 10))

    # Left plot: SDF interior voxels
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(
        normalized_points[:, 0],
        normalized_points[:, 1],
        normalized_points[:, 2],
        c=color_values,
        cmap='viridis',
        s=5,
        alpha=0.3,
        edgecolors='none'
    )
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'SDF Interior Voxels (resolution={resolution}³)')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)

    # Right plot: Normalized mesh
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(
        sampled_mesh[:, 0],
        sampled_mesh[:, 1],
        sampled_mesh[:, 2],
        c='blue',
        s=1,
        alpha=0.5,
        edgecolors='none',
        label='Mesh vertices'
    )
    # Draw the grid bounds
    from itertools import product
    grid_corners = np.array(list(product([-1, 1], [-1, 1], [-1, 1])))
    ax2.scatter(grid_corners[:, 0], grid_corners[:, 1], grid_corners[:, 2],
               c='red', s=100, marker='o', label='Grid bounds [-1,1]³')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Normalized Mesh')
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(-1, 1)
    ax2.legend()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"✓ Saved visualization to {output_path}")

    plt.close()


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
        visualize_sdf(sdf_grid, args.resolution, normalized_mesh, viz_path, verbose=verbose)

    return 0


if __name__ == "__main__":
    exit(main())
