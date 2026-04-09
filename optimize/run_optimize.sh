#!/bin/bash
# Wrapper script to submit a prompt optimization SLURM job with nice CLI args.
# Run from the login node.
#
# Usage:
#   bash optimize/run_optimize.sh --model Qwen/Qwen2.5-1.5B-Instruct
#   bash optimize/run_optimize.sh --model Qwen/Qwen2.5-7B-Instruct --optimizer bootstrap_rs
#   bash optimize/run_optimize.sh --model Qwen/Qwen2.5-1.5B-Instruct --train_size 50 --dev_size 20 --trials 10  # pilot

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
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
OPTIMIZER="mipro"
TRAIN_SIZE=200
DEV_SIZE=100
TRIALS=20
GPU_COUNT=1

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL_NAME="$2";  shift 2 ;;
        --optimizer)    OPTIMIZER="$2";    shift 2 ;;
        --train_size)   TRAIN_SIZE="$2";   shift 2 ;;
        --dev_size)     DEV_SIZE="$2";     shift 2 ;;
        --trials)       TRIALS="$2";       shift 2 ;;
        --gpus)         GPU_COUNT="$2";    shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo ""
            echo "Usage: bash optimize/run_optimize.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL        Model name (default: Qwen/Qwen2.5-1.5B-Instruct)"
            echo "  --optimizer OPT      mipro|bootstrap|bootstrap_rs (default: mipro)"
            echo "  --train_size N       Training examples (default: 200)"
            echo "  --dev_size N         Dev examples (default: 100)"
            echo "  --trials N           Optimization trials (default: 20)"
            echo "  --gpus N             GPU count (default: 2, one for vllm + one for reward model)"
            exit 1
            ;;
    esac
done

# --- Create log directory ---
mkdir -p logs

# --- Print configuration ---
echo "=== Prompt Optimization ==="
echo "Model:     $MODEL_NAME"
echo "Optimizer: $OPTIMIZER"
echo "Train/Dev: $TRAIN_SIZE / $DEV_SIZE"
echo "Trials:    $TRIALS"
echo "GPUs:      $GPU_COUNT"
echo "==========================="
echo ""

# --- Submit job ---
export MODEL_NAME OPTIMIZER TRAIN_SIZE DEV_SIZE TRIALS

JOB_ID=$(sbatch --parsable \
    "${SBATCH_CLUSTER_ARGS[@]}" \
    --gres="gpu:${GPU_COUNT}" \
    --export=ALL \
    slurm/optimize_job.sh)

echo "Submitted optimization job: $JOB_ID"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "View logs:    tail -f logs/mtb-optimize-${JOB_ID}.out"
