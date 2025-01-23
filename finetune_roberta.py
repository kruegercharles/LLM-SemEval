import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import EMOTION_LABELS
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score  # noqa
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)

"""
This script fine-tunes a RoBERTa model for multi-label emotion classification using a given dataset.
It performs the following tasks:
- Configures model and training parameters.
- Loads and preprocesses the dataset.
- Defines metrics for evaluation.
- Fine-tunes the RoBERTa model.
- Evaluates the model and saves the best model and training logs.
- Generates and saves plots for training loss and validation metrics.
"""

# Config
MODEL_TO_FINETUNE_NAME = Path("models/roberta-base/")
CACHE_DIR = Path("cache-dir/")

# Codabench data:
# OUTPUT_DIR = Path("output/roberta-semeval")
# DATA_SET_PATH = Path("data/codabench_data/train/eng_a_parsed.json")

# Emotions data:
# OUTPUT_DIR = Path("output/emotions-data")
# DATA_SET_PATH = Path("data/Emotions_Data/parsed_data.json")

# Dair-Ai Data
# OUTPUT_DIR = Path("output/dair-ai")
# DATA_SET_PATH = Path("data/dair-ai/parsed_data.json")

# GoEmotions data
# OUTPUT_DIR = Path("output/goemotions")
# DATA_SET_PATH = Path("data/go_emotions/parsed_data.json")

# Merged Dataset
OUTPUT_DIR = Path("output/merged-dataset")
DATA_SET_PATH = Path("data/merged_data.json")


# Hyperparameters
LEARNING_RATE: float = 1e-5
EPOCHS: int = 5
BATCH_SIZE: int = 16
CONTEXT_LENGTH: int = 512

tokenizer = RobertaTokenizer.from_pretrained(
    MODEL_TO_FINETUNE_NAME, max_length=CONTEXT_LENGTH
)


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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits.to(device)
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int()
    accuracy = accuracy_score(labels, predictions.numpy())
    f1 = f1_score(labels, predictions.numpy(), average="macro", zero_division="warn")
    return {"accuracy": accuracy, "f1": f1}


def evaluate_metrics(trainer: Trainer):
    train_logs = trainer.state.log_history

    # save logs to file
    with open(str(OUTPUT_DIR) + "/logs.json", "w") as f:
        f.write(json.dumps(train_logs, indent=4))

    best_eval_loss = math.inf
    best_eval_loss_epoch = 0
    best_eval_accuracy = 0.0
    best_eval_accuracy_epoch = 0
    best_eval_f1 = 0.0
    best_eval_f1_epoch = 0

    for log in train_logs:
        if "eval_loss" in log and log["eval_loss"] < best_eval_loss:
            best_eval_loss = log["eval_loss"]
            best_eval_loss_epoch = log["epoch"]
        if "eval_accuracy" in log and log["eval_accuracy"] > best_eval_accuracy:
            best_eval_accuracy = log["eval_accuracy"]
            best_eval_accuracy_epoch = log["epoch"]
        if "eval_f1" in log and log["eval_f1"] > best_eval_f1:
            best_eval_f1 = log["eval_f1"]
            best_eval_f1_epoch = log["epoch"]

    best = []

    best_text_1 = (
        "Best evaluation loss: "
        + str(best_eval_loss)
        + " at epoch "
        + str(best_eval_loss_epoch)
    )
    best.append(best_text_1)
    print("\n" + best_text_1)

    best_text_2 = (
        "Best evaluation accuracy: "
        + str(best_eval_accuracy)
        + " at epoch "
        + str(best_eval_accuracy_epoch)
    )
    best.append(best_text_2)
    print(best_text_2)

    best_text_3 = (
        "Best evaluation f1: "
        + str(best_eval_f1)
        + " at epoch "
        + str(best_eval_f1_epoch)
    )
    best.append(best_text_3)
    print(best_text_3)

    with open(str(OUTPUT_DIR) + "/best.txt", "w") as f:
        f.write("\n".join(best))

    # Separate training and evaluation logs
    training_logs = [log for log in train_logs if "loss" in log]
    eval_logs = [log for log in train_logs if "eval_loss" in log]

    # Extract training loss and epochs
    train_losses = [log["loss"] for log in training_logs]
    train_epochs = [log["epoch"] for log in training_logs]

    # Extract evaluation metrics and epochs
    eval_epochs = [log["epoch"] for log in eval_logs]
    eval_accuracies = [log["eval_accuracy"] for log in eval_logs]
    eval_f1s = [log["eval_f1"] for log in eval_logs]

    eval_loss = [log["eval_loss"] for log in eval_logs]

    # Create the plot
    plt.figure(figsize=(15, 6))

    # Plot training loss
    plt.plot(train_epochs, train_losses, label="Training Loss", color="blue")

    # Plot evaluation metrics (only if there are evaluation logs)
    if eval_logs:
        plt.plot(
            eval_epochs,
            eval_accuracies,
            label="Validation Accuracy",
            color="green",
        )
        plt.plot(
            eval_epochs,
            eval_f1s,
            label="Validation F1 Score (Macro)",
            color="red",
        )
        plt.plot(
            eval_epochs,
            eval_loss,
            label="Validation Loss",
            color="orange",
        )

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Training Loss and Validation Metrics Over Time")
    plt.legend()
    plt.xticks(np.arange(min(train_epochs), max(train_epochs) + 1, 1.0))
    plt.grid(True)
    plt.savefig(str(OUTPUT_DIR) + "/metrics_plot.png")


def load_data():
    dataset = load_dataset("json", data_files=str(DATA_SET_PATH), cache_dir=CACHE_DIR)

    # Split the dataset
    dataset = dataset["train"].train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"].shard(num_shards=2, index=0)
    val_dataset = dataset["test"].shard(num_shards=2, index=1)

    return train_dataset, test_dataset, val_dataset


def preprocess_data(examples):
    # Tokenize the texts
    tokenized: RobertaTokenizer = tokenizer(
        examples["sentence"],
        padding=True,
        truncation=True,
        max_length=CONTEXT_LENGTH,
        return_tensors="pt",  # return as pytorch tensors
    )

    # Convert emotions to labels
    labels = []

    for emotions in examples["emotions"]:
        # Create a multi-hot encoded vector for the emotions

        # set all labels to 0
        label = torch.zeros(len(EMOTION_LABELS))
        for emotion in emotions:
            if emotion not in EMOTION_LABELS:
                print("\n\n=====>>>>>>>>Unknown emotion:", emotion, "\n\n")
                continue
            label[EMOTION_LABELS.index(emotion)] = 1
        labels.append(label)

    tokenized["labels"] = labels

    return tokenized


def finetune():
    print("Cuda is available:", torch.cuda.is_available())
    assert torch.cuda.is_available()

    print_checkpoint("Start finetuning")

    train_dataset, test_dataset, val_dataset = load_data()

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_TO_FINETUNE_NAME,
        num_labels=len(EMOTION_LABELS),
        problem_type="multi_label_classification",
        cache_dir="cache-dir/",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Apply preprocessing to all datasets
    train_dataset = train_dataset.map(
        preprocess_data,
        batched=True,
    )
    test_dataset = test_dataset.map(
        preprocess_data,
        batched=True,
    )
    val_dataset = val_dataset.map(
        preprocess_data,
        batched=True,
    )
    print_checkpoint("Dataset loaded & preprocessed")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        # log_level="info",
        # logging_dir=str(OUTPUT_DIR) + "/logs",
        # logging_steps=10,
        # logging_strategy="steps",
        # gradient_accumulation_steps=8,
        weight_decay=0.01,
        # warmup_steps=500,
        save_strategy="epoch",
        save_total_limit=1,  # only store the best model
        # metric_for_best_model="eval_loss",
        # lr_scheduler_type="cosine",
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        optim="adamw_torch",
    )

    trainer = Trainer(
        args=args,
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print_checkpoint("Training started")
    trainer.train()
    print_checkpoint("Training finished")

    trainer.save_model(str(OUTPUT_DIR))
    print_checkpoint("Model saved")

    evaluate_metrics(trainer)

    # trainer.evaluate()
    # print_checkpoint("Evaluation finished")


def print_checkpoint(message: str):
    print("\n" + "=" * 50)
    print("Checkpoint:", message)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # create_and_clear_folder()
    finetune()
