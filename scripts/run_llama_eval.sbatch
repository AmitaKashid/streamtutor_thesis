#!/bin/bash -l
#SBATCH --partition=gpu7d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Isolated thesis diagnostic; does not touch production jobs or runtime files.
set -euo pipefail
PORT="${1:?Usage: sbatch scripts/run_llama_eval.sbatch PORT}"
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  echo 'Invalid port; use an integer between 1024 and 65535.' >&2
  exit 2
fi
module load gcc/gcc-12.2.0
module load rh8
module load cuda/toolkit-12.4
export LD_LIBRARY_PATH="/beegfs1/software/gcc/gcc-12.2.0/lib64:${LD_LIBRARY_PATH:-}"
SERVER="$HOME/llmtutor/runtimes/llamacpp-b11556/build-gcc12/bin/llama-server"
MODEL="$HOME/llmtutor/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
[[ -x "$SERVER" && -r "$MODEL" ]] || { echo 'Evaluation server executable or exact Llama GGUF is unavailable.' >&2; exit 1; }
printf 'Job: %s\nNode: %s\nModel: %s\nPort: %s\nContext: 8192\nSlots: 1\n' "${SLURM_JOB_ID:-unknown}" "$(hostname -s)" "$MODEL" "$PORT"
printf 'Runtime: %s\nGPU: %s\n' "$SERVER" "${CUDA_VISIBLE_DEVICES:-unset}"
# Compute-node listener, matching the existing cluster model servers.
exec "$SERVER" -m "$MODEL" --host 0.0.0.0 --port "$PORT" \
  -ngl 99 -c 8192 --parallel 1 --jinja --reasoning-format deepseek
