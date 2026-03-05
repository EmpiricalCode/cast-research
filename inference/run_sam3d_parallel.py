#!/usr/bin/env python3
# Parallelized version of run_sam3d.py using torchrun DDP
# Each rank loads its own model on its assigned GPU and processes a subset of masks.
#
# Usage:
#   torchrun --nproc_per_node=NUM_GPUS scripts/run_sam3d_parallel.py [--image ...] [--masks-dir ...] [--output-dir ...]
#
# For multiple processes on a single GPU (time-sliced):
#   torchrun --nproc_per_node=NUM_WORKERS scripts/run_sam3d_parallel.py [args]

import sys
import os
import argparse
import json

import torch
import torch.distributed as dist

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Add sam-3d-objects to path
sam3d_path = os.path.join(project_root, "sam-3d-objects")
sam3d_notebook_path = os.path.join(project_root, "sam-3d-objects", "notebook")
sys.path.insert(0, sam3d_path)
sys.path.insert(0, sam3d_notebook_path)

from inference import Inference, load_image, load_single_mask, make_scene


def main():
    parser = argparse.ArgumentParser(description="Run SAM 3D on masked objects (parallel)")
    parser.add_argument("--image", type=str, default=os.path.join(project_root, "image.jpg"),
                        help="Path to input image")
    parser.add_argument("--masks-dir", type=str, default=os.path.join(project_root, "output/masks"),
                        help="Directory containing mask PNGs")
    parser.add_argument("--output-dir", type=str, default=os.path.join(project_root, "output/sam3d_results"),
                        help="Output directory for 3D results")
    args = parser.parse_args()

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Assign GPU — wraps around if more workers than GPUs
    num_gpus = torch.cuda.device_count()
    device_id = rank % num_gpus if num_gpus > 0 else 0
    torch.cuda.set_device(device_id)

    if rank == 0:
        print(f"World size: {world_size}, GPUs available: {num_gpus}")
        os.makedirs(args.output_dir, exist_ok=True)

    # Wait for rank 0 to create output dir
    dist.barrier()

    # Load model on this rank's GPU
    if rank == 0:
        print("Loading SAM 3D model...")
    tag = "hf"
    config_path = os.path.join(project_root, f"sam-3d-objects/checkpoints/{tag}/pipeline.yaml")
    inference = Inference(config_path, compile=False)

    # Load image (all ranks need it)
    image = load_image(args.image)

    # Get mask list and split across ranks
    mask_files = sorted([f for f in os.listdir(args.masks_dir) if f.endswith('.png')])
    my_masks = mask_files[rank::world_size]  # Round-robin distribution

    if rank == 0:
        print(f"Total masks: {len(mask_files)}, distributing across {world_size} workers")
    print(f"[Rank {rank}] Processing {len(my_masks)} masks: {my_masks}")

    # Process this rank's masks
    metadata = {}
    outputs = []

    for mask_file in my_masks:
        mask_index = int(os.path.splitext(mask_file)[0])
        print(f"[Rank {rank}] Processing mask {mask_index}...")

        mask = load_single_mask(args.masks_dir, index=mask_index, extension=".png")

        try:
            rgba_image = inference.merge_mask_to_rgba(image, mask)
            output = inference._pipeline.run(
                rgba_image,
                None,
                seed=42,
                stage1_only=False,
                with_mesh_postprocess=True,
                with_texture_baking=False,
                with_layout_postprocess=False,
                use_vertex_color=True,
                stage1_inference_steps=None,
                pointmap=None,
            )
        except RuntimeError as e:
            if "numel() == 0" in str(e) or "Expected reduction dim" in str(e):
                print(f"[Rank {rank}] Skipping mask {mask_index}: no valid pointmap data")
                continue
            raise

        outputs.append(output)

        # Save GLB
        output_path = os.path.join(args.output_dir, f"{mask_index}.glb")
        if output.get("glb") is not None:
            output["glb"].export(output_path)
            print(f"[Rank {rank}] Saved {output_path}")
        else:
            print(f"[Rank {rank}] Warning: No mesh for mask {mask_index}")

        metadata[f"{mask_index}"] = {
            "rotation": output["rotation"].cpu().numpy().tolist(),
            "translation": output["translation"].cpu().numpy().tolist(),
            "scale": output["scale"].cpu().numpy().tolist(),
        }

    # Each rank saves its partial metadata to a temp file
    partial_path = os.path.join(args.output_dir, f".positions_rank{rank}.json")
    with open(partial_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Wait for all ranks to finish
    dist.barrier()

    # Rank 0 merges all partial metadata files
    if rank == 0:
        merged_metadata = {}
        for r in range(world_size):
            p = os.path.join(args.output_dir, f".positions_rank{r}.json")
            if os.path.exists(p):
                with open(p, 'r') as f:
                    merged_metadata.update(json.load(f))
                os.remove(p)

        # Sort by key so output is deterministic
        merged_metadata = dict(sorted(merged_metadata.items(), key=lambda x: int(x[0])))

        metadata_path = os.path.join(args.output_dir, "positions.json")
        with open(metadata_path, 'w') as f:
            json.dump(merged_metadata, f, indent=2)
        print(f"\nSaved merged metadata to {metadata_path}")
        print(f"\nAll done! Results saved to {args.output_dir}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
