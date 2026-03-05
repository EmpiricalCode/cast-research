#!/bin/bash
set -e

IMAGE="$1"

if [ -z "$IMAGE" ]; then
    echo "Usage: ./scripts/run_pipeline.sh <image>"
    exit 1
fi

eval "$(conda shell.bash hook)"

echo "=== Step 1: Autolabel ==="
conda activate autoseg
python inference/autolabel.py --tagger qwen --image "$IMAGE"
conda deactivate

echo "=== Step 2: SAM3D ==="
conda activate sam3d-objects
python inference/run_sam3d.py --image "$IMAGE"
conda deactivate

echo "=== Step 3: SDF Optimization ==="
conda activate sdf
PYOPENGL_PLATFORM=egl python3 poc/optimize_sdf.py --dir output/sam3d_results --resolution 32
conda deactivate
mv output/sam3d_results/optimized_positions.json output/sam3d_results/positions.json

echo "=== Step 4: Visualize Scene ==="
conda activate sam3d-objects
python inference/visualize_scene.py
conda deactivate

echo "=== Done ==="
