#!/bin/bash
#SBATCH --job-name=mtb-install
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --time=0-04:00
#SBATCH --gres=gpu:1
#SBATCH --mem=512G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# --- CUDA environment ---
export CUDA_HOME="$HOME/cuda-12.8.1"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# --- Compiler environment ---
source /opt/rh/gcc-toolset-11/enable
export CC=${CC:-$(command -v gcc-11 || command -v gcc)}
export CXX=${CXX:-$(command -v g++-11 || command -v g++)}

# --- Project directory (SLURM_SUBMIT_DIR is where sbatch was called) ---
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"

# --- Activate virtual environment ---
source "$PROJECT_DIR/.venv/bin/activate"

# --- Print environment info ---
echo "=== Install Environment ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Python: $(python --version)"
echo "uv: $(uv --version)"
echo "CUDA_HOME: $CUDA_HOME"
nvidia-smi
echo "=========================="

# --- Install build tools
uv pip install --upgrade pip setuptools wheel cmake ninja

# --- Install dependencies ---
echo "Installing core benchmark dependencies..."
uv sync

# --- Install PyTorch
uv add "torch==2.8.0"

# --- Install vLLM
uv add "opencv-python-headless==4.12.0.88"
uv add --index pypi=https://pypi.org/simple vllm==0.10.2 --extra-index-url https://wheels.vllm.ai/0.10.2/

# --- Verify installation ---
echo "Verifying installation..."
python -c "import openai; import torch; import datasets; import vllm; print(f'All imports OK. torch={torch.__version__}, vllm={vllm.__version__}')"

echo "Installation complete."
