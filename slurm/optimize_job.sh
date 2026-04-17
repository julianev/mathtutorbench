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
MODE="${MODE:-paper}"
# Empty = let run_optimize.py apply the --mode preset.
TRAIN_SIZE="${TRAIN_SIZE:-}"
DEV_SIZE="${DEV_SIZE:-}"
TEST_SIZE="${TEST_SIZE:-}"
AUTO="${AUTO:-light}"

VLLM_PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-600}"
DATA_PATH="${DATA_PATH:-datasets/mathdial_bridge.json}"
OUTPUT="${OUTPUT:-}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

# Optional prompt proposer model (MIPROv2). When set, a second vllm server
# is launched on GPU 1 serving this model. Leave empty to disable.
PROMPT_MODEL_NAME="${PROMPT_MODEL_NAME:-}"
PROMPT_VLLM_PORT="${PROMPT_VLLM_PORT:-8001}"
PROMPT_MAX_MODEL_LEN="${PROMPT_MAX_MODEL_LEN:-32768}"
PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS:-4096}"
PROMPT_TENSOR_PARALLEL_SIZE="${PROMPT_TENSOR_PARALLEL_SIZE:-1}"

# --- CUDA environment (flashinfer in vllm >=0.19 loads libcudart.so.12 via its
#     own CUDA_LIB_PATH env var, defaulting to /usr/local/cuda/... which is
#     permission-locked on this cluster) ---
export CUDA_HOME="$HOME/cuda-12.8.1"
export CUDA_LIB_PATH="$CUDA_HOME/targets/x86_64-linux/lib"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# --- Project directory ---
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"

# --- Activate virtual environment ---
source "$PROJECT_DIR/.venv/bin/activate"

# --- Print job info ---
echo "=== MathTutorBench MIPROv2 Prompt Optimization ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Task model: $MODEL_NAME (port $VLLM_PORT)"
echo "Prompt model: ${PROMPT_MODEL_NAME:-<task model (fallback)>} ${PROMPT_MODEL_NAME:+(port $PROMPT_VLLM_PORT)}"
echo "Mode / auto: $MODE / $AUTO"
echo "Train/Dev/Test: ${TRAIN_SIZE:-<preset>} / ${DEV_SIZE:-<preset>} / ${TEST_SIZE:-<preset>}"
echo "Output: ${OUTPUT:-<auto-timestamped>}"
nvidia-smi
echo "==================================================="

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

# --- Start task vllm server in the background (GPU 0) ---
echo "Starting task vllm server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_NAME" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --seed 42 &
VLLM_PID=$!

# --- Optionally start prompt proposer vllm server on GPU 1 ---
PROMPT_VLLM_PID=""
if [ -n "$PROMPT_MODEL_NAME" ]; then
    echo "Starting prompt proposer vllm server on GPU 1 (model: $PROMPT_MODEL_NAME)..."
    CUDA_VISIBLE_DEVICES=1 vllm serve "$PROMPT_MODEL_NAME" \
        --port "$PROMPT_VLLM_PORT" \
        --tensor-parallel-size "$PROMPT_TENSOR_PARALLEL_SIZE" \
        --max-model-len "$PROMPT_MAX_MODEL_LEN" \
        --seed 42 &
    PROMPT_VLLM_PID=$!
fi

cleanup() {
    echo "Stopping task vllm server (PID: $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    if [ -n "$PROMPT_VLLM_PID" ]; then
        echo "Stopping prompt vllm server (PID: $PROMPT_VLLM_PID)..."
        kill "$PROMPT_VLLM_PID" 2>/dev/null || true
        wait "$PROMPT_VLLM_PID" 2>/dev/null || true
    fi
    echo "vllm servers stopped."
    echo "Cleaning up temp dir: $TMPDIR"
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

# --- Health check (task server, and prompt server if present) ---
wait_for_vllm() {
    local name="$1" port="$2" pid="$3"
    echo "Waiting for $name vllm server on port $port (timeout: ${HEALTH_CHECK_TIMEOUT}s)..."
    local elapsed=0
    local interval=10
    while [ "$elapsed" -lt "$HEALTH_CHECK_TIMEOUT" ]; do
        if curl -s -f "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "$name health check passed after ${elapsed}s."
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: $name vllm server process died unexpectedly."
            exit 1
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    echo "ERROR: $name vllm server did not become ready within ${HEALTH_CHECK_TIMEOUT}s."
    exit 1
}

wait_for_vllm "task" "$VLLM_PORT" "$VLLM_PID"
if [ -n "$PROMPT_VLLM_PID" ]; then
    wait_for_vllm "prompt" "$PROMPT_VLLM_PORT" "$PROMPT_VLLM_PID"
fi

# --- Run optimization (reward model auto-maps to free GPU) ---
echo "Running MIPROv2 prompt optimization..."
OPTIMIZE_ARGS=(
    --model "$MODEL_NAME"
    --base_url "http://localhost:${VLLM_PORT}/v1"
    --max_tokens "$MAX_TOKENS"
    --mode "$MODE"
    --auto "$AUTO"
    --data_path "$DATA_PATH"
)
# Split size flags only included when explicitly set, so --mode preset applies otherwise.
[ -n "$TRAIN_SIZE" ] && OPTIMIZE_ARGS+=(--train_size "$TRAIN_SIZE")
[ -n "$DEV_SIZE" ]   && OPTIMIZE_ARGS+=(--dev_size "$DEV_SIZE")
[ -n "$TEST_SIZE" ]  && OPTIMIZE_ARGS+=(--test_size "$TEST_SIZE")
[ -n "$OUTPUT" ]     && OPTIMIZE_ARGS+=(--output "$OUTPUT")
if [ -n "$PROMPT_MODEL_NAME" ]; then
    OPTIMIZE_ARGS+=(
        --prompt_model "$PROMPT_MODEL_NAME"
        --prompt_base_url "http://localhost:${PROMPT_VLLM_PORT}/v1"
        --prompt_max_tokens "$PROMPT_MAX_TOKENS"
    )
fi

python optimize/run_optimize.py "${OPTIMIZE_ARGS[@]}"

echo "Optimization complete."
echo "Date: $(date)"
