import os  # noqa
import random  # noqa

import torch  # noqa
from torch import Tensor  # noqa
from transformers import (
    RobertaForSequenceClassification,  # noqa
    RobertaTokenizer,
)

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


# Define emotion labels
EMOTION_LABELS = [
    "anger",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "disgust",
]

PROMPT_EXAMPLES = {
    "My pride hurt worse than my leg did.": ["anger", "sadness"],
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

models: list[ModelClass] = []

models.append(ModelClass(name="base-model", path="models/roberta-base/"))  # base model
models.append(
    ModelClass(name="semeval", path="output/roberta-semeval/")
)  # finetuned with codabench data
models.append(
    ModelClass(name="emotions_data", path="output/emotions-data/")
)  # finetuned with emotions data


NUM_ANSWERS = len(models)
print("Number of models:", NUM_ANSWERS)


DEBUG_PRINT_ALL_PROBABILITIES = True

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


TOKENIZER_PATH = "models/roberta-base/"


def prompt():
    tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH, cache_dir="cache-dir/")

    for prompt, solution in PROMPT_EXAMPLES.items():
        print(" ")
        print("-" * 50)
        print(f"Prompt: {prompt}")
        print(" ")

        voting_table = {}
        for emotion in EMOTION_LABELS:
            voting_table[emotion] = 0

        for i, current_model in enumerate(models):
            assert isinstance(current_model, ModelClass)

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
            print(f"Predicted Emotion: {predicted_emotions}")
            print(" ")

        # Go through all results and find the set of emotions that at least NUM_ANSWERS/2 models predicted
        final_answer = []
        for emotion, votes in voting_table.items():
            if votes >= NUM_ANSWERS / 2:
                final_answer.append(emotion)

        print("Voting table:", voting_table)
        print("\nExpected answer:", solution)

        final_answer_text = "Final answer:"

        if not final_answer:
            print(final_answer_text, "none")
        else:
            print(final_answer_text, final_answer)

        evaulate_answer(set(final_answer), set(solution))


def evaulate_answer(answer: set, solution: set):
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

    print(
        "Correct emotions:",
        round((right / (right + wrong) * 100), None),
        "%",
    )


if __name__ == "__main__":
    prompt()
    print(" ")
    print("-" * 50)
    print(" ")
