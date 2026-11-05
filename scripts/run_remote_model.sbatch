#!/bin/bash -l
#SBATCH --partition=gpu7d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/beegfs1/home/ovyj81af/llmtutor/logs/%x_%j.out
#SBATCH --error=/beegfs1/home/ovyj81af/llmtutor/logs/%x_%j.err

set -euo pipefail

MODEL="$1"
PORT="$2"

if [ -z "$MODEL" ] || [ -z "$PORT" ]; then
  echo "Usage: sbatch run_remote_model.sbatch <model> <port>"
  exit 1
fi

# Make sure the module command is available inside sbatch
if ! command -v module >/dev/null 2>&1; then
  source /etc/profile || true
fi

if ! command -v module >/dev/null 2>&1; then
  source /usr/share/Modules/init/bash || true
fi

module load gcc/gcc-12.2.0
module load rh8
module load python/python-3.11.3

cd "$HOME/llmtutor"

source .venv/bin/activate

export LD_LIBRARY_PATH=/beegfs1/software/gcc/gcc-12.2.0/lib64:$HOME/llmtutor/local_lib:/beegfs1/software/python/python-3.11.3/lib:/beegfs1/software/rh8/lib64:/beegfs1/software/rh8/usr/lib64/libibverbs

HOSTNAME_SHORT=$(hostname)

echo "$HOSTNAME_SHORT" > "$HOME/llmtutor/runtime/${MODEL}.host"
echo "$PORT" > "$HOME/llmtutor/runtime/${MODEL}.port"
echo "$SLURM_JOB_ID" > "$HOME/llmtutor/runtime/${MODEL}.jobid"

echo "======================================"
echo "Starting remote LLM server"
echo "Model : $MODEL"
echo "Port  : $PORT"
echo "Node  : $HOSTNAME_SHORT"
echo "JobID : $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-not_set}"
echo "======================================"

which python
python --version
nvidia-smi

python app/remote_llm_server.py \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT"
