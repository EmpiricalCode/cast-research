#!/usr/bin/env python3
"""
Visualize the 3D scene by loading GLB objects, transforming them to world space,
and exporting a colored PLY point cloud.
"""
import argparse
import os
import sys
import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt

# Add project root to path for cast library
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

from cast.correction.transforms import transform_points_quat
from cast.ply import write_ply


def load_object_points(glb_path, num_samples=20000):
    """
    Load mesh from GLB and sample points from it

    Args:
        glb_path: path to GLB file
        num_samples: number of points to sample from mesh

    Returns:
        [N, 3] array of points in local object space
    """
    mesh = trimesh.load(glb_path, force='mesh')

    # Sample points uniformly from mesh surface
    points, _ = trimesh.sample.sample_surface(mesh, num_samples)

    return points


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D scene as colored point cloud PLY")
    parser.add_argument("--dir", type=str, default=os.path.join(project_root, "output/sam3d_results"),
                        help="Directory containing GLB files and positions.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PLY path (default: <dir>/../scene_visualization.ply)")
    args = parser.parse_args()

    # Paths
    sam3d_dir = args.dir
    positions_path = os.path.join(sam3d_dir, "positions.json")

    # Load positions metadata
    print(f"Loading positions from {positions_path}")
    with open(positions_path, 'r') as f:
        positions = json.load(f)

    print(f"Found {len(positions)} objects")

    # Collect all transformed points
    all_points = []
    all_colors = []

    # Color palette for different objects
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, (obj_name, transform) in enumerate(positions.items()):
        # obj_name is now just the index (e.g., "0", "1", "2")
        obj_idx = int(obj_name)
        glb_path = os.path.join(sam3d_dir, f"{obj_idx}.glb")

        if not os.path.exists(glb_path):
            print(f"Warning: {glb_path} not found, skipping")
            continue

        print(f"Loading {obj_name} from {glb_path}")

        # Load points in local space
        local_points = load_object_points(glb_path)

        # Transform to world space
        rotation = transform['rotation']
        translation = transform['translation']
        scale = transform['scale']

        world_points = transform_points_quat(local_points, rotation, translation, scale)

        all_points.append(world_points)

        # Assign color to this object
        color = colors[idx % len(colors)]
        all_colors.append(np.tile(color[:3], (len(world_points), 1)))

        print(f"  Transformed {len(world_points)} points")
        print(f"  Local bounds: [{local_points.min(axis=0)}, {local_points.max(axis=0)}]")
        print(f"  World bounds: [{world_points.min(axis=0)}, {world_points.max(axis=0)}]")
        print(f"  Rotation: {rotation}")
        print(f"  Translation: {translation}")
        print(f"  Scale: {scale}")

    # Concatenate all points
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    print(f"\nTotal points in scene: {len(all_points)}")

    # Export to PLY
    output_path = args.output or os.path.join(os.path.dirname(sam3d_dir), "scene_visualization.ply")
    write_ply(all_points, all_colors, output_path)

    print(f"\nSaved PLY to {output_path} ({len(all_points):,} points)")

if __name__ == "__main__":
    main()
