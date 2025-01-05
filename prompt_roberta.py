import random  # noqa
from collections import Counter  # noqa

import torch  # noqa
from torch import Tensor  # noqa
from transformers import (  # noqa
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
import os  # noqa

# Define emotion labels
EMOTION_LABELS = [
    "anger",
    "fear",
    "joy",
    "sadness",
    "surprise",
]

NUM_ANSWERS = 5
PROMPT_EXAMPLES = {
    "This is a very exciting and happy moment!": ["anger"],
    "Now my parents live in the foothills, and the college is in a large valley.": [
        "none"
    ],
    "I still cannot explain this.": ["fear", "surprise"],
    "Then I decided to try and get up to go to the restroom, but I couldn't move!": [
        "fear",
        "surprise",
    ],
}

# Define threshold for binary classification
THRESHOLD = 0.5


random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# MODEL_NAME = "models/roberta-base/"
MODEL_PATH = "output/roberta-semeval/"
TOKENIZER_PATH = "models/roberta-base/"


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model {MODEL_PATH} not found.")


def prompt():
    for prompt, answer in PROMPT_EXAMPLES.items():
        print(" ")
        print("-" * 50)
        print(f"Prompt: {prompt}")
        print(f"Expected answer: {answer}")
        print(" ")

        voting_table = {}
        for emotion in EMOTION_LABELS:
            voting_table[emotion] = 0

        # all_results = []
        # FIXME: for now the model just gives deterministically the same output each run, except when i load the model each iteration

        for i in range(NUM_ANSWERS):
            # Load model and tokenizer
            tokenizer = RobertaTokenizer.from_pretrained(
                TOKENIZER_PATH, cache_dir="cache-dir/"
            )
            model = RobertaForSequenceClassification.from_pretrained(
                MODEL_PATH,
                num_labels=len(EMOTION_LABELS),
                cache_dir="cache-dir/",
                ignore_mismatched_sizes=True,
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            # Set model to evaluation mode to disable dropout
            model.eval()

            # Tokenize the input
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, padding=True
            ).to(model.device)

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
            # if not predicted_emotions:
            # predicted_emotions = ["none"]

            # all_results.append(predicted_emotions)
            for emotion in predicted_emotions:
                voting_table[emotion] += 1

            print("Probabilities:")
            for label, prob in zip(EMOTION_LABELS, probabilities.squeeze().tolist()):
                print(f"  {label}: {prob:.3f}")
            print(f"==> Predicted Emotion: {predicted_emotions}")
            print(" ")

        # Go through all results and find the set of emotions that at least NUM_ANSWERS/2 models predicted
        final_answer = []
        for emotion, votes in voting_table.items():
            if votes >= NUM_ANSWERS / 2:
                final_answer.append(emotion)
                print(f'Emotion "{emotion}" was predicted by {votes} models.')

        if not final_answer:
            print("Final answer: none")
        else:
            print(f"Final answer: {final_answer}")

        """  most_common = Counter(tuple(x) for x in all_results).most_common(1)
        print(f"Most common emotion set: {most_common[0][0]}") """


if __name__ == "__main__":
    print(" ")
    print("-" * 50)

    prompt()

    print("-" * 50)
    print(" ")
