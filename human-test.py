import os  # noqa
import random  # noqa
import json  # noqa
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa

import torch  # noqa
from torch import Tensor  # noqa
from transformers import (
    RobertaForSequenceClassification,  # noqa
    RobertaTokenizer,
)
from common import EMOTION_LABELS
from sklearn.metrics import accuracy_score, f1_score  # noqa


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
        self.percentage_correct: list[float] = []
        self.percentage_correct_number = 0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.precision = 0
        self.recall = 0
        self.accuracy = 0
        self.f1_score = 0
        self.predictions = []
        self.f1_score_macro = 0
        self.f1_score_weighted = 0


class Human:
    def __init__(self):
        self.name: str = "Human"
        self.percentage_correct: list[float] = []
        self.percentage_correct_number = 0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.precision = 0
        self.recall = 0
        self.accuracy = 0
        self.f1_score = 0
        self.predictions = []
        self.f1_score_macro = 0
        self.f1_score_weighted = 0


def evaulate_answer(
    answer: set, solution: set, model: ModelClass = None, human: Human = None
) -> float:
    """
    Compares the final_answer and the solution and prints the results.
    """

    total = len(solution.union(answer))

    true_negatives = 0
    for element in EMOTION_LABELS:
        if element not in solution and element not in answer:
            true_negatives += 1

    # Every instance that is in both final_answer and solution is correct
    # Every instance that is in final_answer but not in solution is wrong
    # Every instance that is in solution but not in final_answer is wrong

    # get the intersection of the two sets and remove them from both sets
    intersection = answer.intersection(solution)
    true_positives = len(intersection)

    answer -= intersection
    solution -= intersection

    false_positives = len(answer)
    false_negatives = len(solution)

    assert true_positives + false_positives + false_negatives == total

    correct = true_positives / total * 100

    if DEBUG_PRINT_STUFF:
        print(
            "Correct emotions:",
            round(correct, None),
            "%",
        )

    if model is not None:
        model.percentage_correct.append(correct)
        model.tp += true_positives
        model.fp += false_positives
        model.tn += true_negatives
        model.fn += false_negatives

    if human is not None:
        human.percentage_correct.append(correct)
        human.tp += true_positives
        human.fp += false_positives
        human.tn += true_negatives
        human.fn += false_negatives

    return correct


def load_dataset(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)

    dataset = {}
    for dataline in data:
        dataset[dataline["sentence"]] = dataline["emotions"]

    return dataset


def get_precision(tp: int, fp: int) -> float:
    return round(tp / (tp + fp), 2)


def get_recall(tp: int, fn: int) -> float:
    return round(tp / (tp + fn), 2)


def get_f1_score(precision: int, recall: int) -> float:
    return round(2 * (precision * recall) / (precision + recall), 2)


def get_accuracy(tp: int, fp: int, tn: int, fn: int) -> float:
    return round((tp + tn) / (tp + fp + tn + fn), 2)


def get_f1_score_macro(labels: list[list[str]], predictions: list[list[str]]) -> float:
    assert len(labels) == len(predictions)

    f1_scores = []

    possible_emotions = EMOTION_LABELS

    for i, label in enumerate(labels):
        # make sure label and prediction are the same length by comparing them to the possible emotions and adding empty strings
        labels_found = []
        for emotion in possible_emotions:
            if emotion in label:
                labels_found.append(emotion)
            else:
                labels_found.append(" ")

        predictions_found = []
        for emotion in possible_emotions:
            if emotion in predictions[i]:
                predictions_found.append(emotion)
            else:
                predictions_found.append(" ")

        f1_scores.append(f1_score(labels_found, predictions_found, average="macro"))

    return round(sum(f1_scores) / len(f1_scores), 2)


def get_f1_score_weighted(
    labels: list[list[str]], predictions: list[list[str]]
) -> float:
    assert len(labels) == len(predictions)

    f1_scores = []

    possible_emotions = EMOTION_LABELS

    for i, label in enumerate(labels):
        # make sure label and prediction are the same length by comparing them to the possible emotions and adding empty strings
        labels_found = []
        for emotion in possible_emotions:
            if emotion in label:
                labels_found.append(emotion)
            else:
                labels_found.append(" ")

        predictions_found = []
        for emotion in possible_emotions:
            if emotion in predictions[i]:
                predictions_found.append(emotion)
            else:
                predictions_found.append(" ")

        f1_scores.append(f1_score(labels_found, predictions_found, average="weighted"))

    return round(sum(f1_scores) / len(f1_scores), 2)


quit_programm = False


def get_human_feedback(sentence: str) -> list[str]:
    """
    This function is used to get human feedback for the predictions.

    Emotions:
    "anger",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "disgust",
    "none"

    """

    human_answer = []

    print("What emotions do you think are expressed in this sentence?")

    while True:
        human_input = input(
            "[1] anger, [2] fear, [3] joy, [4] sadness, [5] surprise, [6] disgust, [7] none\n[8] done, [9] end programm\n"
        )

        try:
            human_input = int(human_input)

            if human_input < 1 or human_input > 9:
                continue
            if human_input == 8:
                break
            if human_input == 9:
                end = False
                while True:
                    print("Do you really want to end the programm? [y/n]")
                    end_programm = input()
                    if end_programm == "y":
                        end = True
                        break
                    else:
                        break

                if end:
                    global quit_programm
                    quit_programm = True
                    break
            # if answer already in list, continue
            if EMOTION_LABELS[human_input - 1] in human_answer:
                continue

            human_answer.append(EMOTION_LABELS[human_input - 1])

        except ValueError:
            continue

    if len(human_answer) == 0:
        human_answer.append("none")
    print("Human answer:", human_answer)
    return list(set(human_answer))


PROMPT_EXAMPLES: dict = load_dataset("data/codabench_data/dev/eng_a_parsed.json")

# Define threshold for binary classification
THRESHOLD = 0.5

models: list[ModelClass] = []

models.append(
    ModelClass(name="RoBERTa base-model", path="models/roberta-base/")
)  # base model
models.append(
    ModelClass(name="Finetuned semeval", path="output/roberta-semeval/")
)  # finetuned with codabench data
models.append(
    ModelClass(name="Finetuned emotions_data", path="output/emotions-data/")
)  # finetuned with emotions data
models.append(
    ModelClass(name="Finetuned dair-ai", path="output/dair-ai/")
)  # finetuned with dair-ai data
models.append(
    ModelClass(name="Finetuned goemotions", path="output/goemotions/")
)  # finetuned with goemotions data
models.append(
    ModelClass(name="Finetuned merged_dataset", path="output/merged-dataset/")
)  # finetuned with merged dataset


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

    # length = len(PROMPT_EXAMPLES.items())
    # print_checkpoint = length // 10
    # i = 1

    overall_labels = []
    overall_predictions = []

    name = ""

    while name.strip() == "":
        # get name of user via input
        name = input("Please enter your name: ")

    print("Hello", name, "!")

    human = Human()
    human.name = name

    run = 1

    already_seen = []

    while run <= len(PROMPT_EXAMPLES):
        # pick a random prompt
        prompt = random.choice(list(PROMPT_EXAMPLES.keys()))

        # check if prompt was already seen
        if prompt in already_seen:
            continue
        already_seen.append(prompt)

        # get the solution
        solution = PROMPT_EXAMPLES[prompt]

        print("\n\n")
        print(run, '- Sentence:"', prompt, '"')
        run += 1

        overall_labels.append(solution)

        human_answer = get_human_feedback(prompt)

        human.predictions.append(human_answer)
        evaulate_answer(set(human_answer), set(solution), None, human)

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

            current_model.predictions.append(predicted_emotions)

            for emotion in predicted_emotions:
                voting_table[emotion] += 1

            # Create individual statistics for each model
            evaulate_answer(set(predicted_emotions), set(solution), current_model)

        # Go through all results and find the set of emotions that at least half of the models predicted
        final_answer = []
        for emotion, votes in voting_table.items():
            if votes >= len(models) / 2:
                final_answer.append(emotion)

        overall_predictions.append(final_answer)

        final_answer_text = "Answer Modells:"

        if not final_answer:
            print(final_answer_text, "none")
        else:
            print(final_answer_text, final_answer)

        print("Correct answer:", solution)

        statistics_correct_voting_table.append(
            evaulate_answer(set(final_answer), set(solution), None)
        )

        if quit_programm:
            break

    statistics(
        statistics_correct_voting_table, overall_labels, overall_predictions, human
    )


def statistics(
    statistics_correct_voting_table: list[float],
    overall_labels: list,
    overall_predictions: list,
    human: Human,
):
    output_data: list[str] = []

    # Calculate statistics
    print(" ")
    output_data.append("Statistics:")
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    for model in models:
        total_tp += model.tp
        total_fp += model.fp
        total_tn += model.tn
        total_fn += model.fn
    precision = get_precision(tp=total_tp, fp=total_fp)
    recall = get_recall(tp=total_tp, fn=total_fn)
    f1_score = get_f1_score(precision=precision, recall=recall)
    accuracy = get_accuracy(tp=total_tp, fp=total_fp, tn=total_tn, fn=total_fn)
    f1_score_macro = get_f1_score_macro(
        labels=overall_labels, predictions=overall_predictions
    )
    f1_score_weighted = get_f1_score_weighted(
        labels=overall_labels, predictions=overall_predictions
    )
    output_data.append("Modells:")
    output_data.append("Precision overall: " + str(precision))
    output_data.append("Recall overall: " + str(recall))
    output_data.append("Accuracy overall: " + str(accuracy))
    output_data.append("F1-Score overall: " + str(f1_score))
    output_data.append("F1-Score macro overall: " + str(f1_score_macro))
    output_data.append("F1-Score weighted overall: " + str(f1_score_weighted))
    output_data.append(" ")

    output_data.append("Human:")
    output_data.append("Name: " + human.name)
    precision = get_precision(tp=human.tp, fp=human.fp)
    recall = get_recall(tp=human.tp, fn=human.fn)
    f1_score = get_f1_score(precision=precision, recall=recall)
    accuracy = get_accuracy(tp=human.tp, fp=human.fp, tn=human.tn, fn=human.fn)
    f1_score_macro = get_f1_score_macro(
        labels=overall_labels, predictions=human.predictions
    )
    f1_score_weighted = get_f1_score_weighted(
        labels=overall_labels, predictions=human.predictions
    )
    output_data.append("Precision overall: " + str(precision))
    output_data.append("Recall overall: " + str(recall))
    output_data.append("Accuracy overall: " + str(accuracy))
    output_data.append("F1-Score overall: " + str(f1_score))
    output_data.append("F1-Score macro overall: " + str(f1_score_macro))
    output_data.append("F1-Score weighted overall: " + str(f1_score_weighted))

    # Print statistics
    for line in output_data:
        print(line)

    new_data = []
    for line in output_data:
        new_data.append(line + "\n")

    # Read current statistics from file
    with open("human evaluation.txt", "r") as f:
        old_data = f.readlines()

    all_data = old_data + new_data

    # Save statistics to file
    with open("human evaluation.txt", "w") as f:
        f.writelines(all_data)

    """
      plt.figure()

    model_names = [model.name for model in models]
    model_names.append("Overall")
    # precisions = [model.precision for model in models]
    # precisions.append(precision)
    # recalls = [model.recall for model in models]
    # recalls.append(recall)
    f1_scores = [model.f1_score for model in models]
    f1_scores.append(f1_score)
    accuracies = [model.accuracy for model in models]
    accuracies.append(accuracy)
    f1_scores_macro = [model.f1_score_macro for model in models]
    f1_scores_macro.append(f1_score_macro)
    f1_scores_weighted = [model.f1_score_weighted for model in models]
    f1_scores_weighted.append(f1_score_weighted)

    x = np.arange(len(model_names))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))  # adjust figure size for better readability

    # Plot the bars sequentially
    bar_positions = x

    # rects1 = ax.bar(bar_positions, precisions, width, label="Precision")
    # bar_positions = [p + width for p in bar_positions]
    # rects2 = ax.bar(bar_positions, recalls, width, label="Recall")
    bar_positions = [p + width for p in bar_positions]
    rects3 = ax.bar(bar_positions, accuracies, width, label="Accuracy")
    bar_positions = [p + width for p in bar_positions]
    rects4 = ax.bar(bar_positions, f1_scores, width, label="F1-Score")
    bar_positions = [p + width for p in bar_positions]
    rects5 = ax.bar(bar_positions, f1_scores_macro, width, label="F1-Score macro")
    bar_positions = [p + width for p in bar_positions]
    rects6 = ax.bar(bar_positions, f1_scores_weighted, width, label="F1-Score weighted")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Scores")
    ax.set_title("Model Performance Metrics")

    # Adjust x ticks to be in the middle of the bars.
    ax.set_xticks([r + 1.5 * width for r in range(len(model_names))], model_names)
    # ax.set_xticks(x + width*1.5, model_names)

    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    # autolabel(rects1)
    # autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)

    # add labels and title
    plt.xticks(rotation=45)

    plt.legend()
    plt.ylabel("Percentage")
    plt.title("Evaluation Statistics")
    plt.tight_layout(rect=[0.0, 0.03, 1, 1])  # Adjust the bottom margin
    ax.set_ylim(0, 1)

    fig = plt.gcf()
    # save the plot
    fig.savefig("evaluation_statistics_plot.png")
    """


if __name__ == "__main__":
    prompt()
