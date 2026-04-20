#!/bin/bash
set -e

TAGGER="qwen"
PARALLEL=""
NUM_GPUS=2
NO_MOGE=""
SEED=123
LR=0.005
NUM_RESTARTS=200

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tagger) TAGGER="$2"; shift 2 ;;
        --parallel) PARALLEL=1; shift ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --no-moge) NO_MOGE=1; shift ;;
        --seed) SEED="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --num-restarts) NUM_RESTARTS="$2"; shift 2 ;;
        *) IMAGE="$1"; shift ;;
    esac
done

if [ -z "$IMAGE" ]; then
    echo "Usage: ./scripts/run_pipeline.sh [--tagger qwen|gpt] [--parallel] [--num-gpus N] [--no-moge] [--seed N] [--lr F] [--num-restarts N] <image>"
    exit 1
fi

# Initialize conda for activate/deactivate
eval "$(conda shell.bash hook)"

# Generate a unique run directory
RUN_ID=$(python3 -c "import uuid; print(uuid.uuid4().hex[:8])")
RUN_DIR="output/run-${RUN_ID}"
mkdir -p "$RUN_DIR"

echo "=== Run directory: $RUN_DIR ==="

echo "=== Step 1: Autolabel ==="
conda activate autoseg
python inference/autolabel.py --tagger "$TAGGER" --image "$IMAGE" --output-dir "$RUN_DIR"
conda deactivate

echo "=== Step 2: SAM3D ==="
conda activate sam3d-objects
if [ -n "$PARALLEL" ]; then
    python -m torch.distributed.run --nproc_per_node="$NUM_GPUS" inference/run_sam3d_parallel.py --image "$IMAGE" --masks-dir "$RUN_DIR/masks" --output-dir "$RUN_DIR/sam3d_results" --seed "$SEED"
else
    python inference/run_sam3d.py --image "$IMAGE" --masks-dir "$RUN_DIR/masks" --output-dir "$RUN_DIR/sam3d_results" --seed "$SEED"
fi
conda deactivate

if [ -z "$NO_MOGE" ]; then
    echo "=== Step 3: MoGe ICP Refinement ==="
    conda activate moge
    python inference/run_moge.py --image "$IMAGE" --dir "$RUN_DIR"
    conda deactivate
    cp "$RUN_DIR/moge/positions_icp.json" "$RUN_DIR/sam3d_results/positions.json"
else
    echo "=== Step 3: MoGe ICP Refinement (skipped) ==="
fi

echo "=== Step 4: SDF Optimization ==="
conda activate sdf
PYOPENGL_PLATFORM=egl python3 inference/optimize_sdf.py --dir "$RUN_DIR/sam3d_results" --resolution 64 --lr "$LR" --num-restarts "$NUM_RESTARTS"
conda deactivate
mv "$RUN_DIR/sam3d_results/optimized_positions.json" "$RUN_DIR/sam3d_results/positions.json"

echo "=== Step 5: Visualize Scene ==="
conda activate sam3d-objects
python inference/visualize_scene.py --dir "$RUN_DIR/sam3d_results" --output "$RUN_DIR/scene_visualization.ply"

echo "=== Step 6: Export Scene ==="
python inference/export_scene.py --dir "$RUN_DIR/sam3d_results" --output "$RUN_DIR/scene.ply"
conda deactivate

echo "=== Done === Results in $RUN_DIR ==="
