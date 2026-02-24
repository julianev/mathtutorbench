#!/bin/bash
# Master orchestration script for MathTutorBench.
# Submits benchmark and reward model jobs with SLURM dependency chaining.
# Run this from the login node (it is NOT a SLURM job itself).
#
# Usage:
#   bash slurm/run_pipeline.sh
#   bash slurm/run_pipeline.sh --preset qwen-1.5b
#   bash slurm/run_pipeline.sh --preset qwen-7b
#   MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct bash slurm/run_pipeline.sh
#   TASKS=problem_solving.yaml bash slurm/run_pipeline.sh

set -euo pipefail

# --- Project directory ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# --- Cluster config ---
if [ -f slurm/cluster_config.sh ]; then
    source slurm/cluster_config.sh
else
    echo "WARNING: slurm/cluster_config.sh not found."
    echo "Copy slurm/cluster_config.sh.example to slurm/cluster_config.sh and edit for your cluster."
    exit 1
fi

SBATCH_CLUSTER_ARGS=()
[ -n "${PARTITION:-}" ] && SBATCH_CLUSTER_ARGS+=(--partition="$PARTITION")
[ -n "${EXCLUDE:-}" ]   && SBATCH_CLUSTER_ARGS+=(--exclude="$EXCLUDE")

# --- Model presets ---
apply_preset() {
    case "$1" in
        qwen-1.5b)
            export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
            export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
            ;;
        qwen-7b)
            export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
            export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
            ;;
        qwen-32b)
            export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-32B-Instruct}"
            export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
            ;;
        qwen-72b)
            export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-72B-Instruct}"
            export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
            ;;
        *)
            echo "Unknown preset: $1"
            echo "Available presets: qwen-1.5b, qwen-7b, qwen-32b, qwen-72b"
            exit 1
            ;;
    esac
}

# --- Parse arguments ---
PRESET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)
            PRESET="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash slurm/run_pipeline.sh [--preset qwen-1.5b|qwen-7b|qwen-32b|qwen-72b]"
            exit 1
            ;;
    esac
done

if [ -n "$PRESET" ]; then
    apply_preset "$PRESET"
fi

# --- Ensure defaults ---
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

# --- Determine GPU count from tensor parallel size ---
GPU_COUNT="$TENSOR_PARALLEL_SIZE"

# --- Create log directory ---
mkdir -p logs

# --- Print configuration ---
echo "=== MathTutorBench Pipeline ==="
echo "Model: $MODEL_NAME"
echo "GPUs: $GPU_COUNT"
echo "Tensor parallel: $TENSOR_PARALLEL_SIZE"
echo "Tasks: ${TASKS:-all 9 tasks (default)}"
echo "==============================="
echo ""

# --- Submit benchmark job (serve + evaluate) ---
BENCH_JOB_ID=$(sbatch --parsable \
    "${SBATCH_CLUSTER_ARGS[@]}" \
    --gres="gpu:${GPU_COUNT}" \
    --export=ALL \
    slurm/run_benchmark.sh)
echo "Submitted benchmark job: $BENCH_JOB_ID"

# --- Submit reward model job (depends on benchmark completing successfully) ---
REWARD_JOB_ID=$(sbatch --parsable \
    "${SBATCH_CLUSTER_ARGS[@]}" \
    --dependency="afterok:${BENCH_JOB_ID}" \
    --export=ALL \
    slurm/run_reward_model.sh)
echo "Submitted reward model job: $REWARD_JOB_ID (depends on $BENCH_JOB_ID)"

# --- Summary ---
echo ""
echo "=== Pipeline Summary ==="
echo "Benchmark job:    $BENCH_JOB_ID"
echo "Reward model job: $REWARD_JOB_ID (runs after benchmark succeeds)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "View logs:    tail -f logs/*-${BENCH_JOB_ID}.out"
echo "========================"
