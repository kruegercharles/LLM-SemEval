#!/bin/bash

#SBATCH --partition=capella
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/data/walrus/ws/elru535b-llm_emotion/logs/log-%j.log
#SBATCH --error=/data/walrus/ws/elru535b-llm_emotion/logs/errors-%j.log
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=elia.ruehle@mailbox.tu-dresden.de
#SBATCH --account=p_scads_llm_secrets

module purge
module load release/24.04
module load GCCcore/12.3.0
module load Python/3.11.3

source /data/walrus/ws/elru535b-llm_emotion/bin/activate

python3 /data/walrus/ws/elru535b-llm_emotion/LLM-SemEval/src/train.py

deactivate