#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --output=/home/chkr211d/chkr211d-secretllm/project-sem-eval/mylogs/log-%j.log
#SBATCH --error=/home/chkr211d/chkr211d-secretllm/project-sem-eval/mylogs/errors-%j.log
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=charles.krueger@mailbox.tu-dresden.de
#SBATCH --account=p_scads_llm_secrets

source /data/cat/ws/chkr211d-secretllm/bin/activate
cd /home/chkr211d/chkr211d-secretllm/project-sem-eval

python3 finetune_roberta.py
