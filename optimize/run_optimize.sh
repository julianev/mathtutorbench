#!/bin/bash
# Wrapper script to submit a MIPROv2 prompt optimization SLURM job with nice CLI args.
# Run from the login node.
#
# Usage:
#   bash optimize/run_optimize.sh --model Qwen/Qwen3-4B
#   bash optimize/run_optimize.sh --model Qwen/Qwen3-4B \
#       --prompt_model mistralai/Mistral-7B-Instruct-v0.3 --auto medium
#   bash optimize/run_optimize.sh --model Qwen/Qwen3-4B --train_size 50 --dev_size 20  # pilot

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# --- Cluster config ---
if [ -f slurm/cluster_config.sh ]; then
    source slurm/cluster_config.sh
else
    echo "ERROR: slurm/cluster_config.sh not found."
    echo "Copy slurm/cluster_config.sh.example to slurm/cluster_config.sh and edit for your cluster."
    exit 1
fi

SBATCH_CLUSTER_ARGS=()
[ -n "${PARTITION:-}" ] && SBATCH_CLUSTER_ARGS+=(--partition="$PARTITION")
[ -n "${EXCLUDE:-}" ]   && SBATCH_CLUSTER_ARGS+=(--exclude="$EXCLUDE")

# --- Defaults ---
# Leave split sizes empty so the Python script applies the --mode preset.
MODEL_NAME="Qwen/Qwen3-4B"
MODE="paper"
TRAIN_SIZE=""
DEV_SIZE=""
TEST_SIZE=""
AUTO="light"
MAX_TOKENS=2048
PROMPT_MODEL_NAME=""
PROMPT_MAX_TOKENS=4096
PROMPT_MAX_MODEL_LEN=""
GPU_COUNT=""

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)              MODEL_NAME="$2";         shift 2 ;;
        --mode)               MODE="$2";               shift 2 ;;
        --train_size)         TRAIN_SIZE="$2";         shift 2 ;;
        --dev_size)            DEV_SIZE="$2";          shift 2 ;;
        --test_size)          TEST_SIZE="$2";          shift 2 ;;
        --auto)               AUTO="$2";               shift 2 ;;
        --max_tokens)         MAX_TOKENS="$2";         shift 2 ;;
        --prompt_model)          PROMPT_MODEL_NAME="$2";    shift 2 ;;
        --prompt_max_tokens)     PROMPT_MAX_TOKENS="$2";    shift 2 ;;
        --prompt_max_model_len)  PROMPT_MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpus)                  GPU_COUNT="$2";            shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo ""
            echo "Usage: bash optimize/run_optimize.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL             Task model (default: Qwen/Qwen3-4B)"
            echo "  --mode MODE               Split preset: paper|dspy (default: paper)"
            echo "  --train_size N            Override preset: training examples"
            echo "  --dev_size N              Override preset: dev examples"
            echo "  --test_size N             Override preset: held-out test examples"
            echo "  --auto LEVEL              MIPROv2 auto setting: light|medium|heavy (default: light)"
            echo "  --max_tokens N            Task LM completion budget (default: 2048)"
            echo "  --prompt_model MODEL      Stronger model for MIPROv2 instruction proposal"
            echo "                            (served as a second vllm on GPU 1; optional)"
            echo "  --prompt_max_tokens N     Prompt proposer completion budget (default: 4096)"
            echo "  --prompt_max_model_len N  Prompt proposer vllm --max-model-len (default: 32768)"
            echo "                            Lower this (e.g. 16384) for larger proposers (~32B) to leave"
            echo "                            KV-cache headroom on a single 80GB H100."
            echo "  --gpus N                  GPU count (default: 1, or 2 when --prompt_model is set)"
            exit 1
            ;;
    esac
done

# --- Auto-bump GPU count when a separate prompt model is requested ---
if [ -z "$GPU_COUNT" ]; then
    if [ -n "$PROMPT_MODEL_NAME" ]; then
        GPU_COUNT=2
    else
        GPU_COUNT=1
    fi
fi

# --- Create log directory ---
mkdir -p logs

# --- Print configuration ---
echo "=== Prompt Optimization ==="
echo "Model:          $MODEL_NAME"
echo "Prompt model:   ${PROMPT_MODEL_NAME:-(same as task model)}"
echo "Mode:           $MODE"
echo "Train/Dev/Test: ${TRAIN_SIZE:-(preset)} / ${DEV_SIZE:-(preset)} / ${TEST_SIZE:-(preset)}"
echo "Auto:           $AUTO"
echo "Max tokens:     $MAX_TOKENS"
echo "Prompt max-model-len: ${PROMPT_MAX_MODEL_LEN:-(default 32768)}"
echo "GPUs:           $GPU_COUNT"
echo "==========================="
echo ""

# --- Submit job ---
export MODEL_NAME MODE TRAIN_SIZE DEV_SIZE TEST_SIZE AUTO MAX_TOKENS PROMPT_MODEL_NAME PROMPT_MAX_TOKENS PROMPT_MAX_MODEL_LEN

JOB_ID=$(sbatch --parsable \
    "${SBATCH_CLUSTER_ARGS[@]}" \
    --gres="gpu:${GPU_COUNT}" \
    --export=ALL \
    slurm/optimize_job.sh)

echo "Submitted optimization job: $JOB_ID"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "View logs:    tail -f logs/mtb-optimize-${JOB_ID}.out"
