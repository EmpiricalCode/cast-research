#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import os
import argparse

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Add sam-3d-objects to path
sam3d_path = os.path.join(project_root, "sam-3d-objects")
sam3d_notebook_path = os.path.join(project_root, "sam-3d-objects", "notebook")
sys.path.insert(0, sam3d_path)
sys.path.insert(0, sam3d_notebook_path)

from inference import Inference, load_image, load_single_mask, make_scene
import json

def main():
    parser = argparse.ArgumentParser(description="Run SAM 3D on masked objects")
    parser.add_argument("--image", type=str, default=os.path.join(project_root, "image.jpg"),
                        help="Path to input image")
    parser.add_argument("--masks-dir", type=str, default=os.path.join(project_root, "output/masks"),
                        help="Directory containing mask PNGs")
    parser.add_argument("--output-dir", type=str, default=os.path.join(project_root, "output/sam3d_results"),
                        help="Output directory for 3D results")
    args = parser.parse_args()

    # Setup paths
    image_path = args.image
    masks_dir = args.masks_dir
    output_dir = args.output_dir

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load SAM 3D model
    print("Loading SAM 3D model...")
    tag = "hf"
    config_path = os.path.join(project_root, f"sam-3d-objects/checkpoints/{tag}/pipeline.yaml")
    inference = Inference(config_path, compile=False)

    # Load image
    print(f"Loading image from {image_path}...")
    image = load_image(image_path)

    # Get list of masks
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
    print(f"Found {len(mask_files)} masks: {mask_files}")

    # Process each mask
    outputs = []
    metadata = {}

    for mask_file in mask_files:
        mask_index = int(os.path.splitext(mask_file)[0])
        print(f"\nProcessing mask {mask_index}...")

        # Load mask
        mask = load_single_mask(masks_dir, index=mask_index, extension=".png")

        # Run inference with mesh generation enabled
        print(f"  Running SAM 3D inference...")
        try:
            # Merge mask to RGBA
            rgba_image = inference.merge_mask_to_rgba(image, mask)
            # Call pipeline directly to enable mesh generation
            # Note: with_texture_baking=False to avoid requiring diff_gaussian_rasterization
            output = inference._pipeline.run(
                rgba_image,
                None,
                seed=42,
                stage1_only=False,
                with_mesh_postprocess=True,
                with_texture_baking=False,  # Disabled: requires diff_gaussian_rasterization
                with_layout_postprocess=False,
                use_vertex_color=True,  # Use vertex colors instead of texture
                stage1_inference_steps=None,
                pointmap=None,
            )
        except RuntimeError as e:
            if "numel() == 0" in str(e) or "Expected reduction dim" in str(e):
                print(f"  Skipping mask {mask_index}: no valid pointmap data in masked region")
                continue
            raise
        outputs.append(output)

        # Save individual object as GLB mesh
        output_path = os.path.join(output_dir, f"object_{mask_index}.glb")
        if output.get("glb") is not None:
            output["glb"].export(output_path)
            print(f"  Saved 3D mesh to {output_path}")
        else:
            print(f"  Warning: No mesh generated for mask {mask_index}")

        # Save positional metadata
        metadata[f"object_{mask_index}"] = {
            "rotation": output["rotation"].cpu().numpy().tolist(),
            "translation": output["translation"].cpu().numpy().tolist(),
            "scale": output["scale"].cpu().numpy().tolist(),
        }

    # Save metadata JSON
    metadata_path = os.path.join(output_dir, "positions.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved positional metadata to {metadata_path}")

    # Create combined scene with all objects positioned correctly
    if outputs:
        print(f"\nCreating combined scene...")
        scene_gs = make_scene(*outputs)

        # Save as PLY for compatibility
        scene_ply_path = os.path.join(output_dir, "scene_posed.ply")
        scene_gs.save_ply(scene_ply_path)
        print(f"✓ Saved positioned scene (PLY) to {scene_ply_path}")

        # Note: Combined scene as GLB would require mesh extraction from the combined Gaussian splat
        # Individual objects are already saved as GLB files above

    print(f"\n✓ All done! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
