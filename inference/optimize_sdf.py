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
import torch
from pathlib import Path
import json
from dotenv import load_dotenv

# Add project root to path for cast library
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

from cast.correction.sdf import normalize_mesh, compute_sdf_grid, optimize_sdf
from cast.correction.transforms import apply_x90_correction
from cast.correction.relations import extract_relations_from_image

# Load .env - searches current and parent directories automatically
load_dotenv()


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
        target_rotation = apply_x90_correction(target_rotation)

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

    print("\nEXTRACTING RELATIONS FROM IMAGE\n")

    # Use GPT-4o to extract object relations from the output image
    image_path = Path(args.dir).parent / "output.jpg"
    support_relations = {}

    if image_path.exists():
        print(f"Extracting relations from {image_path}...")
        object_ids = sorted(meshes.keys())
        support_relations = extract_relations_from_image(str(image_path), object_ids)
        print(f"Extracted relations: {json.dumps(support_relations, indent=2)}")
    else:
        print(f"Warning: Could not find image ({image_path}), using empty relations")

    print("\nCOMPUTING SDF\n")

    # Compute SDF grids for each mesh (in normal space)
    # Populate sdf_grids dict with [N, N, N] torch tensors representing SDF values
    sdf_grids = {}

    for name, mesh in meshes.items():
        print(f"Computing mesh: {name}")

        normalized_mesh, normalization_scale = normalize_mesh(mesh, target_scale=args.target_scale)

        sdf_grid = compute_sdf_grid(
            normalized_mesh,
            resolution=args.resolution,
        )

        sdf_grids[name] = {
            "grid" : torch.tensor(sdf_grid, dtype=torch.float32),
            "scale" : normalization_scale
        }

    # Run optimization
    print("\nOPTIMIZING POSES\n")
    optimized_transforms = optimize_sdf(sdf_grids, transformations, sampled_points, support_relations)

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
