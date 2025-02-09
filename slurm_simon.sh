#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/data/horse/ws/sicr214c-ces_llm_secret/LLM-SemEval/logs/log-%j.log
#SBATCH --error=/data/horse/ws/sicr214c-ces_llm_secret/LLM-SemEval/logs/errors-%j.log
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=simon.craemer@mailbox.tu-dresden.de
#SBATCH --account=p_scads_llm_secrets

module purge
module load release/24.04 Tcl/8.6.13 OpenPGM/5.2.122 GCC/12.3.0 SQLite/3.43.1 libsodium/1.0.19 OpenSSL/1.1 XZ/5.4.4 util-linux/2.39 GCCcore/13.2.0 libffi/3.4.4 ZeroMQ/4.3.5 zlib/1.2.13 Python/3.11.5 libxml2/2.11.5 binutils/2.40 cffi/1.15.1 libxslt/1.1.38 ncurses/6.4 cryptography/41.0.5 lxml/4.9.3 libreadline/8.2 virtualenv/20.24.6 jedi/0.19.1 bzip2/1.0.8 Python-bundle-PyPI/2023.10 IPython/8.17.2

source /data/horse/ws/sicr214c-ces_llm_secret/venv/bin/activate
srun python3 /data/horse/ws/sicr214c-ces_llm_secret/LLM-SemEval/src/shap_script.py
