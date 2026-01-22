#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import os

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
    # Setup paths
    image_path = os.path.join(project_root, "image.jpg")
    masks_dir = os.path.join(project_root, "output/masks")
    output_dir = os.path.join(project_root, "output/sam3d_results")

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

        # Run inference
        print(f"  Running SAM 3D inference...")
        output = inference(image, mask, seed=42)
        outputs.append(output)

        # Save individual object
        output_path = os.path.join(output_dir, f"object_{mask_index}.ply")
        output["gs"].save_ply(output_path)
        print(f"  Saved 3D reconstruction to {output_path}")

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
    print(f"\nCreating combined scene...")
    scene_gs = make_scene(*outputs)
    scene_path = os.path.join(output_dir, "scene_posed.ply")
    scene_gs.save_ply(scene_path)
    print(f"✓ Saved positioned scene to {scene_path}")

    print(f"\n✓ All done! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
