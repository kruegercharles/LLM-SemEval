import json  # noqa
import os  # noqa
import random  # noqa

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import torch  # noqa
from torch import Tensor  # noqa
from transformers import RobertaForSequenceClassification  # noqa
from transformers import RobertaTokenizer

from common import *  # noqa

# ruff: noqa: F405

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

USE_COMPLEXE_EMOTIONS = False


def select_model(name, backbone, num_labels, device):
    if name == "pure":
        return RobertaForSequenceClassificationPure(backbone, num_labels).to(
            device=device
        )
    elif name == "deep":
        return RobertaForSequenceClassificationDeep(backbone, num_labels).to(
            device=device
        )
    elif name == "mean":
        return RobertaForSequenceClassificationMeanPooling(backbone, num_labels).to(
            device=device
        )
    elif name == "max":
        return RobertaForSequenceClassificationMaxPooling(backbone, num_labels).to(
            device=device
        )
    elif name == "attention":
        return RobertaForSequenceClassificationAttentionPooling(
            backbone, num_labels
        ).to(device=device)
    else:
        raise ValueError("Specified model name is not available!")


class ModelClass:
    def __init__(self, name: str, path: str, num_labels: int):
        self.name: str = name
        self.path: str = path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model {path} not found.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = select_model(
            name=name, backbone=path, num_labels=num_labels, device=device
        )
        self.model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True), strict=False
        )

        self.model.eval()

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
        self.labels = []
        self.predictions = []
        self.f1_score_macro = 0
        self.f1_score_weighted = 0


def evaluate_answer(answer: set, solution: set, model: ModelClass = None) -> float:
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

    return correct


def load_dataset(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)

    dataset = {}
    for dataline in data:
        dataset[dataline["sentence"]] = dataline["emotions"]

    return dataset


print("Start script")


# Define threshold for binary classification
THRESHOLD = 0.5


if USE_COMPLEXE_EMOTIONS:
    PROMPT_EXAMPLES = load_dataset("data/codabench_data/dev/eng_b_parsed.json")
else:
    PROMPT_EXAMPLES = load_dataset("data/codabench_data/dev/eng_a_parsed.json")

models: list[ModelClass] = []

if USE_COMPLEXE_EMOTIONS:
    models.append(
        ModelClass(name="SemEval complexe", path="output/roberta-semeval-complexe/")
    )
else:
    models.append(
        ModelClass(
            name="pure",
            path="output/pure/",
            num_labels=len(PROMPT_EXAMPLES.items()),
        )
    )
    models.append(
        ModelClass(
            name="deep",
            path="output/deep/",
            num_labels=len(PROMPT_EXAMPLES.items()),
        )
    )
    models.append(
        ModelClass(
            name="mean",
            path="output/mean/",
            num_labels=len(PROMPT_EXAMPLES.items()),
        )
    )
    models.append(
        ModelClass(
            name="max",
            path="output/max/",
            num_labels=len(PROMPT_EXAMPLES.items()),
        )
    )
    models.append(
        ModelClass(
            name="attention",
            path="output/attention/",
            num_labels=len(PROMPT_EXAMPLES.items()),
        )
    )

    # models.append(
    # ModelClass(name="RoBERTa base-model", path="models/roberta-base/")
    # )  # base model
    # models.append(
    # ModelClass(name="Finetuned emotions_data", path="output/emotions-data/")
    # )  # finetuned with emotions data
    # models.append(
    # ModelClass(name="Finetuned dair-ai", path="output/dair-ai/")
    # )  # finetuned with dair-ai data
    # models.append(
    # ModelClass(name="Finetuned goemotions", path="output/goemotions/")
    # )  # finetuned with goemotions data
    # models.append(
    # ModelClass(name="Finetuned merged_dataset", path="output/merged-dataset/")
    # )  # finetuned with merged dataset """


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

    length = len(PROMPT_EXAMPLES.items())
    print_checkpoint = length // 10
    i = 1

    overall_labels = []
    overall_predictions = []

    for prompt, solution in PROMPT_EXAMPLES.items():
        if i % print_checkpoint == 0:
            print(f"Progress: {i}/{length}")
        i += 1

        if DEBUG_PRINT_STUFF:
            print(" ")
            print("-" * 50)
            print(f"Prompt: {prompt}")
            print(" ")

        overall_labels.append(solution)

        voting_table = {}

        if USE_COMPLEXE_EMOTIONS:
            for emotion in EMOTION_COMPLEX_LABELS:
                voting_table[emotion] = 0
        else:
            for emotion in EMOTION_LABELS:
                voting_table[emotion] = 0

        for run, current_model in enumerate(models):
            assert isinstance(current_model, ModelClass)

            if DEBUG_PRINT_STUFF:
                print("Run:", run + 1, "with model:", current_model.name)

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
            if USE_COMPLEXE_EMOTIONS:
                predicted_emotions = [
                    EMOTION_COMPLEX_LABELS[j]
                    for j, val in enumerate(predicted_labels)
                    if val == 1
                ]
            else:
                predicted_emotions = [
                    EMOTION_LABELS[j]
                    for j, val in enumerate(predicted_labels)
                    if val == 1
                ]

            if len(predicted_emotions) == 0:
                predicted_emotions.append("none")
            if len(predicted_emotions) > 1 and "none" in predicted_emotions:
                predicted_emotions.remove("none")

            assert (
                len(predicted_emotions) > 0
            )  # There must always be at least one emotion predicted
            assert ("none" not in predicted_emotions) or (len(predicted_emotions) == 1)

            current_model.labels.append(solution)
            current_model.predictions.append(predicted_emotions)

            for emotion in predicted_emotions:
                voting_table[emotion] += 1

            # Create individual statistics for each model
            evaluate_answer(set(predicted_emotions), set(solution), current_model)

            if DEBUG_PRINT_ALL_PROBABILITIES:
                print("Probabilities:")
                if USE_COMPLEXE_EMOTIONS:
                    for label, prob in zip(
                        EMOTION_COMPLEX_LABELS, probabilities.squeeze().tolist()
                    ):
                        print(f"  {label}: {prob:.3f}")
                else:
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

        if len(final_answer) == 0:
            final_answer.append("none")
        if len(final_answer) > 1 and "none" in final_answer:
            votes_for_none = voting_table["none"]
            max_votes_other = 0
            for emotion, votes in voting_table.items():
                if emotion != "none" and votes > max_votes_other:
                    max_votes_other = votes
            if max_votes_other >= votes_for_none:
                final_answer.remove("none")
            else:
                final_answer = ["none"]

        assert (
            len(final_answer) > 0
        )  # There must always be at least one emotion predicteds
        assert ("none" not in final_answer) or (len(final_answer) == 1)

        overall_predictions.append(final_answer)

        if DEBUG_PRINT_STUFF:
            print("Voting table:", voting_table)
            print("\nExpected answer:", solution)

            final_answer_text = "Final answer:"

            if not final_answer:
                print(final_answer_text, "none")
            else:
                print(final_answer_text, final_answer)

        statistics_correct_voting_table.append(
            evaluate_answer(set(final_answer), set(solution), None)
        )

    statistics(statistics_correct_voting_table, overall_labels, overall_predictions)


def statistics(
    statistics_correct_voting_table: list[float],
    overall_labels: list,
    overall_predictions: list,
):
    output_data: list[str] = []

    # Calculate statistics
    print(" ")
    output_data.append("Statistics:")
    percentage = sum(statistics_correct_voting_table) / len(
        statistics_correct_voting_table
    )
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
    f1_score_macro = get_f1_score_macro(overall_labels, overall_predictions)
    f1_score_weighted = get_f1_score_weighted(overall_labels, overall_predictions)
    output_data.append("Precision overall: " + str(precision))
    output_data.append("Recall overall: " + str(recall))
    output_data.append("Accuracy overall: " + str(accuracy))
    output_data.append("F1-Score overall: " + str(f1_score))
    output_data.append("F1-Score macro overall: " + str(f1_score_macro))
    output_data.append("F1-Score weighted overall: " + str(f1_score_weighted))

    output_data.append("Average correct emotions: " + str(round(percentage, 2)) + "%")
    output_data.append(" ")
    for model in models:
        output_data.append(f"Model '{model.name}':")
        model.percentage_correct_number = sum(model.percentage_correct) / len(
            model.percentage_correct
        )
        output_data.append(
            "Percentage correct: "
            + str(round(model.percentage_correct_number, 2))
            + "%",
        )
        model.precision = get_precision(tp=model.tp, fp=model.fp)
        model.recall = get_recall(tp=model.tp, fn=model.fn)
        model.f1_score = get_f1_score(precision=model.precision, recall=model.recall)
        model.accuracy = get_accuracy(
            tp=model.tp, fp=model.fp, tn=model.tn, fn=model.fn
        )
        model.f1_score_macro = get_f1_score_macro(model.labels, model.predictions)
        model.f1_score_weighted = get_f1_score_weighted(model.labels, model.predictions)
        output_data.append("Precision: " + str(model.precision))
        output_data.append("Recall: " + str(model.recall))
        output_data.append("F1-Score: " + str(model.f1_score))
        output_data.append("Accuracy: " + str(model.accuracy))
        output_data.append("F1-Score macro: " + str(model.f1_score_macro))
        output_data.append("F1-Score weighted: " + str(model.f1_score_weighted))
        output_data.append(" ")

    # Print statistics
    for line in output_data:
        print(line)

    # Save statistics to file
    if USE_COMPLEXE_EMOTIONS:
        with open("evaluation_statistics_text_complexe.txt", "w") as f:
            for line in output_data:
                f.write(str(line) + "\n")
    else:
        with open("evaluation_statistics_text.txt", "w") as f:
            for line in output_data:
                f.write(str(line) + "\n")

    plt.figure()

    model_names = [model.name for model in models]
    # model_names.append("Overall")
    # precisions = [model.precision for model in models]
    # precisions.append(precision)
    # recalls = [model.recall for model in models]
    # recalls.append(recall)
    # f1_scores = [model.f1_score for model in models]
    # f1_scores.append(f1_score)
    accuracies = [model.accuracy for model in models]
    # accuracies.append(accuracy)
    f1_scores_macro = [model.f1_score_macro for model in models]
    # f1_scores_macro.append(f1_score_macro)
    f1_scores_weighted = [model.f1_score_weighted for model in models]
    # f1_scores_weighted.append(f1_score_weighted)

    x = np.arange(len(model_names))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))  # adjust figure size for better readability

    # Plot the bars sequentially
    bar_positions = x

    # rects1 = ax.bar(bar_positions, precisions, width, label="Precision")
    # bar_positions = [p + width for p in bar_positions]
    # rects2 = ax.bar(bar_positions, recalls, width, label="Recall")
    # bar_positions = [p + width for p in bar_positions]
    rects3 = ax.bar(bar_positions, accuracies, width, label="Accuracy")
    bar_positions = [p + width for p in bar_positions]
    # rects4 = ax.bar(bar_positions, f1_scores, width, label="F1-Score")
    # bar_positions = [p + width for p in bar_positions]
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
        """Attach a text label above each bar in *rects*, displaying its height."""
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
    # autolabel(rects4)
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

    if USE_COMPLEXE_EMOTIONS:
        fig.savefig("evaluation_statistics_plot_complexe.png")
    else:
        fig.savefig("evaluation_statistics_plot.png")


if __name__ == "__main__":
    prompt()
    if DEBUG_PRINT_STUFF:
        print(" ")
        print("-" * 50)
        print(" ")
