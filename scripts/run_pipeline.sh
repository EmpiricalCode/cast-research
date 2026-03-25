#!/bin/bash
set -e

TAGGER="qwen"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tagger) TAGGER="$2"; shift 2 ;;
        *) IMAGE="$1"; shift ;;
    esac
done

if [ -z "$IMAGE" ]; then
    echo "Usage: ./scripts/run_pipeline.sh [--tagger qwen|gpt] <image>"
    exit 1
fi

# Generate a unique run directory
RUN_ID=$(python3 -c "import uuid; print(uuid.uuid4().hex[:8])")
RUN_DIR="output/run-${RUN_ID}"
mkdir -p "$RUN_DIR"

echo "=== Run directory: $RUN_DIR ==="

echo "=== Step 1: Autolabel ==="
conda run --no-banner -n autoseg python inference/autolabel.py --tagger "$TAGGER" --image "$IMAGE" --output-dir "$RUN_DIR"

echo "=== Step 2: SAM3D ==="
conda run --no-banner -n sam3d-objects python inference/run_sam3d.py --image "$IMAGE" --masks-dir "$RUN_DIR/masks" --output-dir "$RUN_DIR/sam3d_results"

echo "=== Step 3: SDF Optimization ==="
conda run --no-banner -n sdf env PYOPENGL_PLATFORM=egl python3 inference/optimize_sdf.py --dir "$RUN_DIR/sam3d_results" --resolution 64
mv "$RUN_DIR/sam3d_results/optimized_positions.json" "$RUN_DIR/sam3d_results/positions.json"

echo "=== Step 4: Visualize Scene ==="
conda run --no-banner -n sam3d-objects python inference/visualize_scene.py --dir "$RUN_DIR/sam3d_results" --output "$RUN_DIR/scene_visualization.ply"

echo "=== Done === Results in $RUN_DIR ==="
