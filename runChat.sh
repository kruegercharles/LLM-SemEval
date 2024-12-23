
# check if the environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Please activate your Python virtual environment before running this script."
    exit 1
fi

# run
time torchrun --nproc_per_node gpu main.py --ckpt_dir models/checkpoints/Llama3.1-8B-Instruct --tokenizer_path models/checkpoints/Llama3.1-8B-Instruct/tokenizer.model