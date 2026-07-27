#!/bin/bash
#SBATCH --job-name=llmtutor_run
#SBATCH --partition=gpu7d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "===== JOB START ====="
date
hostname

# ---- Load modules ----
module load gcc/gcc-12.2.0
module load rh8
module load python/python-3.11.3

# ---- Activate project ----
cd ~/llmtutor
source .venv/bin/activate

# ---- Fix libraries ----
export LD_LIBRARY_PATH=/beegfs1/software/gcc/gcc-12.2.0/lib64:$HOME/llmtutor/local_lib:/beegfs1/software/python/python-3.11.3/lib:/beegfs1/software/rh8/lib64:/beegfs1/software/rh8/usr/lib64/libibverbs

# ---- Copy model to local disk (IMPORTANT) ----
mkdir -p /tmp/$USER/llmtutor_models
cp -u ~/llmtutor/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf /tmp/$USER/llmtutor_models/

# ---- Debug checks ----
echo "===== ENV CHECK ====="
python -c "import sounddevice; print('sounddevice ok')"
python -c "import llama_cpp; print('llama_cpp ok')"
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('count:', torch.cuda.device_count())"

echo "===== RUNNING PIPELINE ====="

# ---- Run your pipeline ----
python app/mic_stt_llm_tts.py \
  --input stt/sample.wav \
  --llm_model llama3_8b_q4 \
  --stt_device cuda \
  --stt_model medium \
  --stt_beam_size 5 \
  --stt_compute_type float16 \
  --mode stream-chunked \
  --tts_backend kokoro \
  --rag \
  --rag_top_k 2 \
  --teaching_style socratic \
  --save

echo "===== JOB END ====="
date