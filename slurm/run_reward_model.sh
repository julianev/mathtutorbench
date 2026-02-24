#!/bin/bash
#SBATCH --job-name=mtb-reward
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=0-01:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Runs the pedagogical reward model on benchmark generation outputs.
# Evaluates scaffolding quality by comparing model-generated teacher
# utterances against ground truth using a 1.5B reward model.
#
# Usage:
#   sbatch slurm/run_reward_model.sh
#   OUTPUT_DIR=results sbatch slurm/run_reward_model.sh

set -euo pipefail

# --- Configurable parameters ---
OUTPUT_DIR="${OUTPUT_DIR:-results}"

# --- Project directory (SLURM_SUBMIT_DIR is where sbatch was called) ---
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"

# --- Activate virtual environment ---
source "$PROJECT_DIR/.venv/bin/activate"

# --- Print job info ---
echo "=== MathTutorBench Reward Model ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Output dir: $OUTPUT_DIR"
nvidia-smi
echo "===================================="

# --- Find and process generation files ---
GENERATION_FILES=$(find "$OUTPUT_DIR" -name "generations-*.json" -type f)

if [ -z "$GENERATION_FILES" ]; then
    echo "ERROR: No generation files found in $OUTPUT_DIR/"
    echo "Run the benchmark first to generate teacher utterances."
    exit 1
fi

echo "Found generation files:"
echo "$GENERATION_FILES"
echo ""

for DATA_PATH in $GENERATION_FILES; do
    echo "--- Evaluating: $DATA_PATH ---"
    python reward_model/compute_scaffolding_score.py --data_path "$DATA_PATH"
    echo ""
done

echo "Reward model evaluation complete."
echo "Date: $(date)"
