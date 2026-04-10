#!/bin/bash
#SBATCH --job-name=mtb-optimize
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --time=0-08:00
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Internal SLURM job script for prompt optimization.
# Submitted by slurm/run_optimize.sh — don't call directly.

set -euo pipefail

# --- Parameters (passed via environment from run_optimize.sh) ---
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
OPTIMIZER="${OPTIMIZER:-mipro}"
TRAIN_SIZE="${TRAIN_SIZE:-200}"
DEV_SIZE="${DEV_SIZE:-100}"
TRIALS="${TRIALS:-20}"
AUTO="${AUTO:-light}"

VLLM_PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-600}"
DATA_PATH="${DATA_PATH:-datasets/mathdial_bridge.json}"
OUTPUT="${OUTPUT:-}"

# --- Project directory ---
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"

# --- Activate virtual environment ---
source "$PROJECT_DIR/.venv/bin/activate"

# --- Print job info ---
echo "=== MathTutorBench Prompt Optimization ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Model: $MODEL_NAME"
echo "Optimizer: $OPTIMIZER"
echo "Train/Dev/Trials: $TRAIN_SIZE / $DEV_SIZE / $TRIALS"
echo "Port: $VLLM_PORT"
echo "Output: $OUTPUT"
nvidia-smi
echo "==========================================="

# --- Writable temp directory (some nodes restrict /scratch_local) ---
mkdir -p "${PROJECT_DIR}/.tmp"
TMPDIR="$(mktemp -d "${PROJECT_DIR}/.tmp/optimize-XXXXXX")"
export TMPDIR
# Redirect torch/triton caches into the job-specific TMPDIR to avoid stale
# references to /scratch_local dirs from previous SLURM jobs.
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torch_inductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
echo "Temp dir: $TMPDIR"

# --- Clear stale vllm compile cache (may contain hardcoded paths from old jobs) ---
rm -rf "${HOME}/.cache/vllm/torch_compile_cache"

# --- Start vllm server in the background (GPU 0) ---
echo "Starting vllm server..."
vllm serve "$MODEL_NAME" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --seed 42 &
VLLM_PID=$!

cleanup() {
    echo "Stopping vllm server (PID: $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    echo "vllm server stopped."
    echo "Cleaning up temp dir: $TMPDIR"
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

# --- Health check ---
echo "Waiting for vllm server to be ready (timeout: ${HEALTH_CHECK_TIMEOUT}s)..."
ELAPSED=0
INTERVAL=10
while [ "$ELAPSED" -lt "$HEALTH_CHECK_TIMEOUT" ]; do
    if curl -s -f "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "Health check passed after ${ELAPSED}s."
        break
    fi

    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vllm server process died unexpectedly."
        exit 1
    fi

    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ "$ELAPSED" -ge "$HEALTH_CHECK_TIMEOUT" ]; then
    echo "ERROR: vllm server did not become ready within ${HEALTH_CHECK_TIMEOUT}s."
    exit 1
fi

# --- Run optimization (reward model auto-maps to free GPU) ---
echo "Running prompt optimization..."
OPTIMIZE_ARGS=(
    --model "$MODEL_NAME"
    --base_url "http://localhost:${VLLM_PORT}/v1"
    --optimizer "$OPTIMIZER"
    --train_size "$TRAIN_SIZE"
    --dev_size "$DEV_SIZE"
    --trials "$TRIALS"
    --auto "$AUTO"
    --data_path "$DATA_PATH"
)
[ -n "$OUTPUT" ] && OPTIMIZE_ARGS+=(--output "$OUTPUT")

python optimize/run_optimize.py "${OPTIMIZE_ARGS[@]}"

echo "Optimization complete."
echo "Date: $(date)"
