#!/bin/bash
# Login-node wrapper to submit slurm/install_env.sh with cluster-specific args.
# Run from the login node:
#   bash slurm/install.sh

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

mkdir -p logs

JOB_ID=$(sbatch --parsable \
    "${SBATCH_CLUSTER_ARGS[@]}" \
    slurm/install_env.sh)
echo "Submitted install job: $JOB_ID"
echo "Logs: logs/mtb-install-${JOB_ID}.{out,err}"
