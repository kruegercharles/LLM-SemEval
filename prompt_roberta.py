import os  # noqa
import random  # noqa
import json  # noqa

import torch  # noqa
from torch import Tensor  # noqa
from transformers import (
    RobertaForSequenceClassification,  # noqa
    RobertaTokenizer,
)
from common import EMOTION_LABELS


"""
This script performs emotion classification using an ensemble of RoBERTa models.

It loads a pre-trained RoBERTa model fine-tuned for emotion classification.
For each input prompt, it runs the model multiple times.
It uses a voting system to aggregate the predictions from each run.
The final predicted emotions are those that receive at least half the total votes.
The script compares the final predictions with expected answers for evaluation.

Key aspects:
- Uses a pre-trained RoBERTa model.
- Employs an ensemble method with voting.
- Predicts multiple emotions for each prompt.
- Evaluates predictions against expected answers.
- Currently loads the same model multiple times, which limits the ensemble's effectiveness.
"""


class ModelClass:
    def __init__(self, name: str, path: str):
        self.name: str = name
        self.path: str = path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model {path} not found.")

        self.model: RobertaForSequenceClassification = (
            RobertaForSequenceClassification.from_pretrained(
                path,
                num_labels=len(EMOTION_LABELS),
                cache_dir="cache-dir/",
                ignore_mismatched_sizes=True,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        )


def evaulate_answer(answer: set, solution: set) -> float:
    """
    Compares the final_answer and the solution and prints the results.
    """
    right = 0
    wrong = 0

    total = len(solution.union(answer))

    # Every instance that is in both final_answer and solution is correct
    # Every instance that is in final_answer but not in solution is wrong
    # Every instance that is in solution but not in final_answer is wrong

    # get the intersection of the two sets and remove them from both sets
    intersection = answer.intersection(solution)
    right += len(intersection)
    answer -= intersection
    solution -= intersection

    union = answer.union(solution)
    wrong += len(union)

    assert right + wrong == total

    correct = right / total * 100

    if DEBUG_PRINT_STUFF:
        print(
            "Correct emotions:",
            round(correct, None),
            "%",
        )

    return correct


def load_dataset(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)

    dataset = {}
    for dataline in data:
        dataset[dataline["sentence"]] = dataline["emotions"]

    return dataset


PROMPT_EXAMPLES = load_dataset("data/codabench_data/dev/eng_a_parsed.json")

# Define threshold for binary classification
THRESHOLD = 0.5

models: list[ModelClass] = []

models.append(ModelClass(name="base-model", path="models/roberta-base/"))  # base model
models.append(
    ModelClass(name="semeval", path="output/roberta-semeval/")
)  # finetuned with codabench data
models.append(
    ModelClass(name="emotions_data", path="output/emotions-data/")
)  # finetuned with emotions data


DEBUG_PRINT_ALL_PROBABILITIES = False
DEBUG_PRINT_STUFF = False

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


TOKENIZER_PATH = "models/roberta-base/"


def prompt():
    tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH, cache_dir="cache-dir/")

    statistics_correct_voting_table: list[float] = []

    for prompt, solution in PROMPT_EXAMPLES.items():
        if DEBUG_PRINT_STUFF:
            print(" ")
            print("-" * 50)
            print(f"Prompt: {prompt}")
            print(" ")

        voting_table = {}
        for emotion in EMOTION_LABELS:
            voting_table[emotion] = 0

        for i, current_model in enumerate(models):
            assert isinstance(current_model, ModelClass)

            if DEBUG_PRINT_STUFF:
                print("Run:", i + 1, "with model:", current_model.name)

            # Set model to evaluation mode to disable dropout
            current_model.model.eval()

            # Tokenize the input
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, padding=True
            ).to(current_model.model.device)

            # Perform inference
            with torch.no_grad():
                outputs: Tensor = current_model.model(**inputs)

            # Get predicted probabilities
            probabilities = torch.sigmoid(outputs.logits)

            # Apply threshold
            predicted_labels = (probabilities > THRESHOLD).int().squeeze().tolist()

            # Get predicted emotions
            predicted_emotions = [
                EMOTION_LABELS[j] for j, val in enumerate(predicted_labels) if val == 1
            ]

            for emotion in predicted_emotions:
                voting_table[emotion] += 1

            if DEBUG_PRINT_ALL_PROBABILITIES:
                print("Probabilities:")
                for label, prob in zip(
                    EMOTION_LABELS, probabilities.squeeze().tolist()
                ):
                    print(f"  {label}: {prob:.3f}")
            if DEBUG_PRINT_STUFF:
                print(f"Predicted Emotion: {predicted_emotions}")
                print(" ")

        # Go through all results and find the set of emotions that at least half of the models predicted
        final_answer = []
        for emotion, votes in voting_table.items():
            if votes >= len(models) / 2:
                final_answer.append(emotion)

        if DEBUG_PRINT_STUFF:
            print("Voting table:", voting_table)
            print("\nExpected answer:", solution)

            final_answer_text = "Final answer:"

            if not final_answer:
                print(final_answer_text, "none")
            else:
                print(final_answer_text, final_answer)

        statistics_correct_voting_table.append(
            evaulate_answer(set(final_answer), set(solution))
        )

    # Calculate statistics
    print(" ")
    print("Statistics:")
    percentage = sum(statistics_correct_voting_table) / len(
        statistics_correct_voting_table
    )
    print("Average correct emotions:", round(percentage, 2), "%")
    print(" ")


if __name__ == "__main__":
    prompt()
    if DEBUG_PRINT_STUFF:
        print(" ")
        print("-" * 50)
        print(" ")
