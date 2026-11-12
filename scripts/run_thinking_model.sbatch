#!/bin/bash -l
#SBATCH --partition=gpu7d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

MODEL_KEY="${1:?Usage: sbatch run_thinking_model.sbatch MODEL_KEY PORT}"
PORT="${2:?Usage: sbatch run_thinking_model.sbatch MODEL_KEY PORT}"
case "$MODEL_KEY" in
  gemma4_e4b_thinking) MODEL_FILE="$HOME/llmtutor/models/gemma-4-E4B_q4_0-it.gguf" ;;
  bonsai2_9b_thinking) MODEL_FILE="$HOME/llmtutor/models/Ternary-Bonsai-2-9B-Q2_0.gguf" ;;
  *) echo "Unknown thinking model: $MODEL_KEY" >&2; exit 2 ;;
esac
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  echo 'Invalid server port' >&2; exit 2
fi

module load gcc/gcc-12.2.0
module load rh8
module load cuda/toolkit-12.4
export LD_LIBRARY_PATH="/beegfs1/software/gcc/gcc-12.2.0/lib64:${LD_LIBRARY_PATH:-}"
SERVER="$HOME/llmtutor/runtimes/llamacpp-b11556/build-gcc12/bin/llama-server"
[[ -x "$SERVER" && -r "$MODEL_FILE" ]] || { echo 'Server or model file unavailable' >&2; exit 1; }

echo "Job: ${SLURM_JOB_ID}"
echo "Node: $(hostname -s)"
echo "Model: $MODEL_KEY"
echo "Port: $PORT"
echo 'Starting llama-server; check /health for readiness.'
# This binds the compute node network interface, matching the current remote
# tutor server. The port must only be reachable within the trusted cluster.
exec "$SERVER" -m "$MODEL_FILE" --host 0.0.0.0 --port "$PORT" \
  -ngl 99 -c 8192 --jinja --reasoning-format deepseek
