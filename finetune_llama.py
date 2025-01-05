from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers import LlamaForCausalLM
import os

# Config
MODEL_PATH = "models/checkpoints/Llama3.1-8B-Instruct/"
OUTPUT_DIR = "output/meta-llama-semeval"
DATA_SET_PATH = "data/codabench_data/train/eng_a_parsed.json"


def finetune(
    pretrained_model_name_or_path: Path,
    output_directory: Path,
    learning_rate: float = 3e-4,  # 5e-4,
    epochs: int = 5,
    batch_size: int = 32,
    context_length: int = 512,  # 128,
    cache_dir: Path = "cache-dir/",
):
    print_checkpoint("Start finetuning")
    # remove folder output
    OUTPUT_DIR = Path("output/")

    def recursive_delete(directory):
        if not os.path.exists(directory):
            print(f"Directory '{directory}' does not exist.")
            return

        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)

                if os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"Deleted file: {item_path}")
                elif os.path.isdir(item_path):
                    recursive_delete(item_path)  # Recursively delete subdirectories
                    os.rmdir(item_path)  # Delete the now empty subdirectory
                    print(f"Deleted directory: {item_path}")

            # Check if the directory is now empty (important for the initial call)
            if not os.listdir(directory):
                os.rmdir(directory)
                print(f"Deleted directory: {directory}")
        except OSError as e:
            print(f"Error deleting {directory}: {e}")

    recursive_delete(OUTPUT_DIR)

    # create folder output
    output_directory.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("json", data_files=DATA_SET_PATH, cache_dir=cache_dir)
    dataset_list = []
    for data in dataset["train"]:
        emotions = str(data["emotions"])
        emotions = emotions.replace("[", "")
        emotions = emotions.replace("]", "")
        emotions = emotions.replace("'", "")

        formatted_data = (
            f"USER: Sentence: {data['sentence']}\nSYSTEM: Emotions: {emotions}"
        )
        dataset_list.append(formatted_data)
    print("\nExample data:")
    print(dataset_list[0])
    print(dataset_list[1])
    print(dataset_list[2])
    print_checkpoint("Dataset loaded")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    print_checkpoint("Tokenizer loaded")

    train_dataset = tokenizer(
        dataset_list,
        add_special_tokens=True,
        truncation=True,
        max_length=context_length,
    )["input_ids"]
    print("\nExample tokenized data:")
    print(train_dataset[0])
    print(train_dataset[1])
    print(train_dataset[2])

    print_checkpoint("Dataset tokenized")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    print_checkpoint("Data collator created")

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        device_map="auto",
        # quantization_config=config,
        use_cache=False,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    print_checkpoint("Model loaded")

    # # Training setup
    # if hasattr(model, "gradient_checkpointing_enable"):
    #     model.gradient_checkpointing_enable()

    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # trainable_percentage = (trainable_params / total_params) * 100

    # print("Total parameters:", total_params)
    # print("Trainable parameters:", trainable_params)
    # print("Trainable percentage: {:.2f}%".format(trainable_percentage))

    args = TrainingArguments(
        output_dir=output_directory,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        gradient_accumulation_steps=8,
        logging_steps=5_000,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=learning_rate,
        save_steps=5_000,
        fp16=False,
    )

    trainer = Trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    print_checkpoint("Training started")

    print(f"Model size: {model.num_parameters()/1000**3:.1f}B parameters")
    trainer.train()
    print_checkpoint("Training finished")

    trainer.save_model(output_directory)
    print_checkpoint("Model saved")


def print_checkpoint(message: str):
    # just a helper function to make the output more readable when reaching a checkpoint
    print(" ")
    print("=" * 50)
    print("Checkpoint", message)
    print("=" * 50)
    print(" ")


if __name__ == "__main__":
    pretrained_model = Path(MODEL_PATH)
    output_directory = Path(OUTPUT_DIR)

    finetune(pretrained_model, output_directory)
    # instruct(pretrained_model, output_directory)
    # prompt(output_directory)
