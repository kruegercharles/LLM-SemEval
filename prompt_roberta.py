import random  # noqa
from collections import Counter  # noqa

import torch  # noqa
from torch import Tensor  # noqa
from transformers import (  # noqa
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

# Define emotion labels
EMOTION_LABELS = [
    "anger",
    "fear",
    "joy",
    "sadness",
    "surprise",
]

NUM_ANSWERS = 3
PROMPT = "This is a very exciting and happy moment!"

# Define threshold for binary classification
THRESHOLD = 0.5


random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# Load model and tokenizer
MODEL_NAME = "models/roberta-base/"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(EMOTION_LABELS), cache_dir="cache-dir"
).to("cuda" if torch.cuda.is_available() else "cpu")

# Set model to evaluation mode to disable dropout
model.eval()

# Tokenize the input
inputs = tokenizer(PROMPT, return_tensors="pt", truncation=True, padding=True).to(
    model.device
)


def main():
    print(" ")
    print("-" * 50)
    print(f"Prompt: {PROMPT}")

    all_results = []

    # FIXME: for now the model just gives deterministically the same output each run
    for i in range(NUM_ANSWERS):
        print("Run:", i + 1)

        # Perform inference
        with torch.no_grad():
            outputs: Tensor = model(**inputs)

        # Get predicted probabilities
        probabilities = torch.sigmoid(outputs.logits)

        # Apply threshold
        predicted_labels = (probabilities > THRESHOLD).int().squeeze().tolist()

        # Get predicted emotions
        predicted_emotions = [
            EMOTION_LABELS[j] for j, val in enumerate(predicted_labels) if val == 1
        ]
        if not predicted_emotions:
            predicted_emotions = ["none"]

        all_results.append(predicted_emotions)

        print("Probabilities:")
        for label, prob in zip(EMOTION_LABELS, probabilities.squeeze().tolist()):
            print(f"  {label}: {prob:.3f}")
        print(f"==> Predicted Emotion: {predicted_emotions}")
        print(" ")

    most_common = Counter(tuple(x) for x in all_results).most_common(1)
    print(f"Most common emotion set: {most_common[0][0]}")
    print("-" * 50)
    print(" ")


if __name__ == "__main__":
    main()
