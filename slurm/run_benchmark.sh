#!/bin/bash
#SBATCH --job-name=mtb-benchmark
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --time=0-04:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Serves a model with vllm and runs the MathTutorBench evaluation on the same node.
#
# GPU memory guidance:
#   Qwen2.5-1.5B-Instruct:  1 GPU,  ~4GB  VRAM
#   Qwen2.5-7B-Instruct:    1 GPU,  ~16GB VRAM
#   Qwen2.5-32B-Instruct:   4 GPUs, ~80GB VRAM (set TENSOR_PARALLEL_SIZE=4, --gres=gpu:4)
#   Qwen2.5-72B-Instruct:   4+ GPUs,       VRAM (set TENSOR_PARALLEL_SIZE=4, --gres=gpu:4)
#
# Usage:
#   sbatch slurm/run_benchmark.sh
#   MODEL_NAME=Qwen/Qwen2.5-7B-Instruct sbatch slurm/run_benchmark.sh
#   TASKS=problem_solving.yaml sbatch slurm/run_benchmark.sh

set -euo pipefail

# --- Configurable parameters (override via environment variables) ---
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TASKS="${TASKS:-problem_solving.yaml,socratic_questioning.yaml,student_solution_correctness.yaml,mistake_location.yaml,mistake_correction.yaml,scaffolding_generation.yaml,pedagogy_following.yaml,scaffolding_generation_hard.yaml,pedagogy_following_hard.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-600}"

# --- Project directory (SLURM_SUBMIT_DIR is where sbatch was called) ---
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"

# --- Activate virtual environment ---
source "$PROJECT_DIR/.venv/bin/activate"

# --- Print job info ---
echo "=== MathTutorBench Benchmark ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Model: $MODEL_NAME"
echo "Tasks: $TASKS"
echo "Port: $VLLM_PORT"
echo "Tensor parallel: $TENSOR_PARALLEL_SIZE"
echo "Max model len: $MAX_MODEL_LEN"
echo "Output dir: $OUTPUT_DIR"
nvidia-smi
echo "================================"

# --- Start vllm server in the background ---
echo "Starting vllm server..."
vllm serve "$MODEL_NAME" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --seed 42 &
VLLM_PID=$!

# Ensure vllm is killed when this script exits (success, error, or signal)
cleanup() {
    echo "Stopping vllm server (PID: $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    echo "vllm server stopped."
}
trap cleanup EXIT

# --- Health check: wait for vllm to be ready ---
echo "Waiting for vllm server to be ready (timeout: ${HEALTH_CHECK_TIMEOUT}s)..."
ELAPSED=0
INTERVAL=10
while [ "$ELAPSED" -lt "$HEALTH_CHECK_TIMEOUT" ]; do
    if curl -s -f "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "Health check passed after ${ELAPSED}s."
        break
    fi

    # Check if vllm process is still alive
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

# --- Run benchmark ---
echo "Running benchmark..."
python main.py \
    --tasks "$TASKS" \
    --provider completion_api \
    --model_args "base_url=http://localhost:${VLLM_PORT}/v1,model=${MODEL_NAME}" \
    --output "$OUTPUT_DIR"

echo "Benchmark complete. Results saved to $OUTPUT_DIR/"
echo "Date: $(date)"
