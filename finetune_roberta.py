import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    DataCollatorWithPadding,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)

# Config
MODEL_NAME = Path("models/roberta-base/")
OUTPUT_DIR = Path("output/roberta-semeval")
DATA_SET_PATH = Path("data/codabench_data/train/eng_a_parsed.json")
CACHE_DIR = Path("cache-dir/")

# Hyperparameters
LEARNING_RATE: float = 3e-4
EPOCHS: int = 5
BATCH_SIZE: int = 32
CONTEXT_LENGTH: int = 512

EMOTION_LABELS = [
    "anger",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "none",
]


def create_and_clear_folder():
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
                    recursive_delete(item_path)
                    os.rmdir(item_path)
                    print(f"Deleted directory: {item_path}")

            if not os.listdir(directory):
                os.rmdir(directory)
                print(f"Deleted directory: {directory}")
        except OSError as e:
            print(f"Error deleting {directory}: {e}")

    recursive_delete(OUTPUT_BASE_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def convert_emotions_to_labels(emotions):
    # Create a multi-hot encoded vector for the emotions
    label = torch.zeros(len(EMOTION_LABELS))
    for emotion in emotions:
        if emotion in EMOTION_LABELS:
            label[EMOTION_LABELS.index(emotion)] = 1
    return label.tolist()


def finetune():
    print_checkpoint("Start finetuning")

    dataset = load_dataset("json", data_files=str(DATA_SET_PATH), cache_dir=CACHE_DIR)

    # Split the dataset
    dataset = dataset["train"].train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"].shard(num_shards=2, index=0)
    val_dataset = dataset["test"].shard(num_shards=2, index=1)

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        # Tokenize the texts
        tokenized: RobertaTokenizer = tokenizer(
            examples["sentence"],
            padding=True,
            truncation=True,
            max_length=CONTEXT_LENGTH,
            return_tensors=None,
        )

        # Convert emotions to labels
        labels = [
            convert_emotions_to_labels(emotions) for emotions in examples["emotions"]
        ]
        tokenized["labels"] = labels

        return tokenized

    # Apply preprocessing to all datasets
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=test_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    print("Features:", train_dataset.features)
    # Output: Features: {'input_ids': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None), 'attention_mask': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None), 'labels': Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None)}

    num_labels = dataset["train"].features["label"].num_classes
    class_names = dataset["train"].features["label"].names
    print(f"Number of labels: {num_labels}")
    print(f"The labels: {class_names}")

    print_checkpoint("Dataset loaded & preprocessed")

    # Set format for PyTorch
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    val_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True, return_tensors="pt"
    )

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(EMOTION_LABELS),
        cache_dir="cache-dir/",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_dir=str(OUTPUT_DIR) + "/logs",
        logging_steps=10,
        logging_strategy="steps",
        gradient_accumulation_steps=8,
        weight_decay=0.01,
        warmup_steps=500,
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
    )

    trainer = Trainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print_checkpoint("Training started")
    trainer.train()
    print_checkpoint("Training finished")

    trainer.evaluate()
    print_checkpoint("Evaluation finished")

    trainer.save_model(str(OUTPUT_DIR))
    print_checkpoint("Model saved")


def print_checkpoint(message: str):
    print("\n" + "=" * 50)
    print("Checkpoint:", message)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    create_and_clear_folder()
    finetune()
