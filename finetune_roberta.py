import os
from pathlib import Path

import torch  # noqa
import torch.nn as nn  # noqa
from datasets import load_dataset
from transformers import (
    RobertaForSequenceClassification,  # noqa
    RobertaTokenizer,  # noqa
    Trainer,  # noqa
    TrainingArguments,  # noqa
)

# Config
MODEL_NAME = Path("models/roberta-base/")
OUTPUT_DIR = Path("output/roberta-semeval")
DATA_SET_PATH = Path("data/codabench_data/train/eng_a_parsed.json")
CACHE_DIR = Path(
    "cache-dir/",
)

# Hyperparameters
LEARNING_RATE: float = 3e-4  # 5e-4,
EPOCHS: int = 5
BATCH_SIZE: int = 32
CONTEXT_LENGTH: int = 512  # 128,

EMOTION_LABELS = [
    "anger",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "none",
]


def create_and_clear_folder():
    # remove folder output
    OUTPUT_BASE_DIR = Path("output/")

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

    recursive_delete(OUTPUT_BASE_DIR)

    # create folder output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def finetune():
    print_checkpoint("Start finetuning")

    dataset = load_dataset("json", data_files=str(DATA_SET_PATH), cache_dir=CACHE_DIR)

    # Split the dataset into train and test datasets
    dataset = dataset["train"].train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"].shard(num_shards=2, index=0)
    eval_dataset = dataset["test"].shard(num_shards=2, index=1)

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding=True,
            truncation=True,
            max_length=CONTEXT_LENGTH,
        )

    train_dataset = train_dataset.map(
        tokenize_function, batched=True, batch_size=len(train_dataset)
    )
    test_dataset = test_dataset.map(
        tokenize_function, batched=True, batch_size=len(test_dataset)
    )
    eval_dataset = eval_dataset.map(
        tokenize_function, batched=True, batch_size=len(eval_dataset)
    )

    print_checkpoint("Dataset loaded & tokenized")

    # set format
    train_dataset.set_format("torch", columns=["input_ids", "sentence", "emotions"])
    test_dataset.set_format("torch", columns=["input_ids", "sentence", "emotions"])
    eval_dataset.set_format("torch", columns=["input_ids", "sentence", "emotions"])
    print("train_dataset:", train_dataset)
    print("test_dataset:", test_dataset)
    print("val_dataset:", eval_dataset)

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(EMOTION_LABELS), cache_dir="cache-dir/"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    """  dataset_list = []
    for data in dataset["train"]:
        emotions = str(data["emotions"])
        emotions = emotions.replace("[", "")
        emotions = emotions.replace("]", "")
        emotions = emotions.replace("'", "")

        formatted_data = (
            f"USER: Sentence: {data['sentence']}\nSYSTEM: Emotions: {emotions}"
        )
        # FIXME: das ist bullshit

        dataset_list.append(formatted_data)

    print("\nExample data:")
    print(dataset_list[0])
    print(dataset_list[1])
    print(dataset_list[2])
    print_checkpoint("Dataset loaded")

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

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

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(EMOTION_LABELS), cache_dir="cache-dir/"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    print_checkpoint("Model loaded")
 """

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_dir=str(OUTPUT_DIR) + "/logs",
        gradient_accumulation_steps=8,
        logging_steps=50,
        weight_decay=0.01,
        warmup_steps=500,
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print_checkpoint("Training started")

    trainer.train()
    print_checkpoint("Training finished")

    trainer.evaluate()
    print_checkpoint("Evaluation finished")

    trainer.save_model(str(OUTPUT_DIR))
    print_checkpoint("Model saved")


def print_checkpoint(message: str):
    # just a helper function to make the output more readable when reaching a checkpoint
    print(" ")
    print("=" * 50)
    print("Checkpoint", message)
    print("=" * 50)
    print(" ")


if __name__ == "__main__":
    create_and_clear_folder()
    finetune()
