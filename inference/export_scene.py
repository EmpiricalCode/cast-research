#!/usr/bin/env python3
"""
Export a combined PLY mesh by loading GLB objects, transforming them
to world space using positions.json, and merging into a single file.
"""
import argparse
import os
import sys
import json
import numpy as np
import trimesh

# Add project root to path for cast library
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

from cast.correction.transforms import transform_points_quat


def main():
    parser = argparse.ArgumentParser(description="Export combined PLY mesh from GLB objects")
    parser.add_argument("--dir", type=str, default=os.path.join(project_root, "output/sam3d_results"),
                        help="Directory containing GLB files and positions.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PLY path (default: <dir>/../scene.ply)")
    args = parser.parse_args()

    # Paths
    sam3d_dir = args.dir
    positions_path = os.path.join(sam3d_dir, "positions.json")

    # Load positions metadata
    print(f"Loading positions from {positions_path}")
    with open(positions_path, 'r') as f:
        positions = json.load(f)

    print(f"Found {len(positions)} objects")

    meshes = []

    for obj_name, transform in positions.items():
        obj_idx = int(obj_name)
        glb_path = os.path.join(sam3d_dir, f"{obj_idx}.glb")

        if not os.path.exists(glb_path):
            print(f"Warning: {glb_path} not found, skipping")
            continue

        print(f"Loading {obj_name} from {glb_path}")

        # Load mesh, convert texture to vertex colors before concatenating
        scene = trimesh.load(glb_path)
        geoms = []
        for geom in scene.geometry.values():
            geom.visual = geom.visual.to_color()
            geoms.append(geom)
        mesh = trimesh.util.concatenate(geoms)

        # Transform vertices to world space
        rotation = transform['rotation']
        translation = transform['translation']
        scale = transform['scale']

        world_vertices = transform_points_quat(mesh.vertices, rotation, translation, scale)
        mesh.vertices = world_vertices

        print(f"  {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        meshes.append(mesh)

    # Combine all meshes
    combined = trimesh.util.concatenate(meshes)
    print(f"\nCombined: {len(combined.vertices)} vertices, {len(combined.faces)} faces")

    # Export as PLY
    output_path = args.output or os.path.join(os.path.dirname(sam3d_dir), "scene.ply")
    combined.export(output_path, file_type='ply')
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
