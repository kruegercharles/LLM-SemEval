#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=48G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --output=/data/cat/ws/elru535b-llm_secrets/logs/log-%j.log
#SBATCH --error=/data/cat/ws/elru535b-llm_secrets/logs/errors-%j.log
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=elia.ruehle@mailbox.tu-dresden.de
#SBATCH --account=p_scads_llm_secrets

module purge
module load release/24.04
module load GCCcore/12.3.0
module load Python/3.11.3

source /data/cat/ws/elru535b-llm_secrets/bin/activate

srun python3 /data/cat/ws/elru535b-llm_secrets/LLM-SemEval/src/train.py