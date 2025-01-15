#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/data/walrus/ws/elru535b-llm_emotion/logs/log-%j.log
#SBATCH --error=/data/walrus/ws/elru535b-llm_emotion/logs/errors-%j.log
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=elia.ruehle@mailbox.tu-dresden.de
#SBATCH --account=p_scads_llm_secrets


source /data/walrus/ws/elru535b-llm_emotion/bin/activate

python3 /data/walrus/ws/elru535b-llm_emotion/LLM-SemEval/src/train.py

deactivate