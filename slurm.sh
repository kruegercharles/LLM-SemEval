#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/data/walrus/ws/elru535b-llm_emotion/logs/log-%j.log
#SBATCH --error=/data/walrus/ws/elru535b-llm_emotion/logs/errors-%j.log
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=elia.ruehle@mailbox.tu-dresden.de
#SBATCH --account=p_scads_llm_secrets

source /data/walrus/ws/elru535b-llm_emotion/bin/activate

python3 train.py

deactivate