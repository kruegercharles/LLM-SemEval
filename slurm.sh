#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/data/cat/ws/chkr211d-secretllm/mylogs/log-%j.log
#SBATCH --error=/data/cat/ws/chkr211d-secretllm/mylogs/errors-%j.log
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=charles.krueger@mailbox.tu-dresden.de
#SBATCH --account=p_scads_llm_secrets

python3 -m venv --system-site-package /data/cat/ws/chkr211d-secretllm
source /data/cat/ws/chkr211d-secretllm/bin/activate

torchrun --nproc_per_node 1 example_text_completion.py --ckpt_dir models/checkpoints/Llama3.1-8B-Instruct --tokenizer_path models/checkpoints/Llama3.1-8B-Instruct/tokenizer.model --max_seq_len 128 --max_batch_size 4
