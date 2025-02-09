import json  # noqa
import os  # noqa
import random  # noqa

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import torch  # noqa
from torch import Tensor  # noqa
from transformers import RobertaForSequenceClassification  # noqa
from transformers import RobertaTokenizer

import copy  # noqa

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


random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

USE_COMPLEXE_EMOTIONS = False


class ModelClass:
    def __init__(
        self,
        name: str,
        path: str,
    ):
        self.name: str = name
        self.path: str = path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model {path} not found.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_labels = len(EMOTION_LABELS)

        self.model = select_model(
            name=name, backbone="roberta-base", num_labels=num_labels, device=device
        )
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()

        stat_item = {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }
        self.stats = {}
        for emotion in EMOTION_LABELS:
            self.stats[emotion] = copy.deepcopy(stat_item)

        self.precision = 0
        self.recall = 0
        self.accuracy = 0
        self.f1_score_micro = 0
        self.f1_score_macro = 0
        self.f1_score_weighted = 0


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
            path="output/pure/RobertaForSequenceClassificationPure_fold_3_epoch_5.pth",
        )
    )
    models.append(
        ModelClass(
            name="deep",
            path="output/deep/RobertaForSequenceClassificationDeep_fold_2_epoch_8.pth",
        )
    )
    models.append(
        ModelClass(
            name="mean",
            path="output/mean/RobertaForSequenceClassificationMeanPooling_fold_5_epoch_8.pth",
        )
    )
    models.append(
        ModelClass(
            name="max",
            path="output/max/RobertaForSequenceClassificationMaxPooling_fold_3_epoch_8.pth",
        )
    )
    models.append(
        ModelClass(
            name="attention",
            path="output/attention/RobertaForSequenceClassificationAttentionPooling_fold_5_epoch_7.pth",
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


TOKENIZER_PATH = "models/roberta-base/"


def prompt():
    tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH, cache_dir="cache-dir/")

    length = len(PROMPT_EXAMPLES.items())
    print_checkpoint = length // 10
    i = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for prompt, solution in PROMPT_EXAMPLES.items():
        if i % print_checkpoint == 0:
            print(f"Progress: {i}/{length}")
        i += 1

        voting_table = {}

        if USE_COMPLEXE_EMOTIONS:
            for emotion in EMOTION_COMPLEX_LABELS:
                voting_table[emotion] = 0
        else:
            for emotion in EMOTION_LABELS:
                voting_table[emotion] = 0

        for current_model in models:
            assert isinstance(current_model, ModelClass)

            # Tokenize the input
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, padding=True
            ).to(device)

            # Perform inference
            with torch.no_grad():
                outputs: Tensor = current_model.model(**inputs)

            # Get predicted probabilities
            probabilities = torch.sigmoid(outputs)

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

            for emotion in predicted_emotions:
                voting_table[emotion] += 1

            # Create individual statistics for each model
            for emotion in EMOTION_LABELS:
                if emotion in predicted_emotions and emotion in solution:
                    current_model.stats[emotion]["tp"] += 1
                elif emotion in predicted_emotions and emotion not in solution:
                    current_model.stats[emotion]["fp"] += 1
                elif emotion not in predicted_emotions and emotion in solution:
                    current_model.stats[emotion]["fn"] += 1
                elif emotion not in predicted_emotions and emotion not in solution:
                    current_model.stats[emotion]["tn"] += 1
                else:
                    raise ValueError("Invalid state")

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


def validate():
    # check if the values makes sence
    total = len(PROMPT_EXAMPLES.items())
    for model in models:
        for emotion in EMOTION_LABELS:
            sum = (
                model.stats[emotion]["tp"]
                + model.stats[emotion]["fp"]
                + model.stats[emotion]["tn"]
                + model.stats[emotion]["fn"]
            )
            if sum != total:
                print("Model: ", model.name)
                print("State: ", model.stats)
                print("Emotion: ", emotion)
                print("Total: ", total)
                print("Sum: ", sum)
                raise ValueError("Invalid state")


def statistics():
    output_data: list[str] = []

    # Calculate statistics
    print(" ")
    output_data.append("Statistics:")

    for model in models:
        print(model.name, model.stats)

        output_data.append(f"Model '{model.name}':")

        accuracies = []
        precisions = []
        recalls = []
        f1_scores_macro = []
        support = []
        f1_scores_weighted = []

        tps = 0
        tns = 0
        fps = 0
        fns = 0

        for emotion in EMOTION_LABELS:
            tp = model.stats[emotion]["tp"]
            tps += tp
            fp = model.stats[emotion]["fp"]
            fps += fp
            tn = model.stats[emotion]["tn"]
            tns += tn
            fn = model.stats[emotion]["fn"]
            fns += fn

            f1 = get_f1_score(tp=tp, fp=fp, fn=fn)

            f1_scores_macro.append(f1)
            accuracies.append(get_accuracy(tp=tp, fp=fp, tn=tn, fn=fn))
            precisions.append(get_precision(tp=tp, fp=fp))
            recalls.append(get_recall(tp=tp, fn=fn))

            support.append(tp + fn)
            f1_scores_weighted.append(f1 * (tp + fn))

        model.precision = round(sum(precisions) / len(precisions), 2)
        model.recall = round(sum(recalls) / len(recalls), 2)
        model.accuracy = round(sum(accuracies) / len(accuracies), 2)
        model.f1_score_micro = round(get_f1_score(tp=tps, fp=fps, fn=fns), 2)
        model.f1_score_macro = round(sum(f1_scores_macro) / len(f1_scores_macro), 2)
        model.f1_score_weighted = round(sum(f1_scores_weighted) / sum(support), 2)

        output_data.append("Precision: " + str(model.precision))
        output_data.append("Recall: " + str(model.recall))
        output_data.append("F1-Score micro: " + str(model.f1_score_micro))
        output_data.append("F1-Score macro: " + str(model.f1_score_macro))
        output_data.append("F1-Score weighted: " + str(model.f1_score_weighted))
        output_data.append("Accuracy: " + str(model.accuracy))
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
    # accuracies = [model.accuracy for model in models]
    f1_scores_micro = [model.f1_score_micro for model in models]
    f1_scores_macro = [model.f1_score_macro for model in models]
    # f1_scores_weighted = [model.f1_score_weighted for model in models]

    model_names.insert(0, "Baseline")
    f1_scores_micro.insert(0, 0.74)
    f1_scores_macro.insert(0, 0.65)

    x = np.arange(len(model_names))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))  # adjust figure size for better readability

    # Plot the bars sequentially
    bar_positions = x

    # rects3 = ax.bar(bar_positions, accuracies, width, label="Accuracy")
    # bar_positions = [p + width for p in bar_positions]
    rects4 = ax.bar(bar_positions, f1_scores_micro, width, label="F1-Score micro")
    bar_positions = [p + width for p in bar_positions]
    rects5 = ax.bar(bar_positions, f1_scores_macro, width, label="F1-Score macro")
    # bar_positions = [p + width for p in bar_positions]
    # rects6 = ax.bar(bar_positions, f1_scores_weighted, width, label="F1-Score weighted")

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
    # autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    # autolabel(rects6)

    # add labels and title
    plt.xticks(rotation=45)

    plt.legend()
    plt.ylabel("Percentage")
    plt.title("Result of the Evaluation with the SemEval evaluation dataset")
    plt.tight_layout(rect=[0.0, 0.03, 1, 1])  # Adjust the bottom margin
    ax.set_ylim(0, 1)

    fig = plt.gcf()

    # save the plot
    if USE_COMPLEXE_EMOTIONS:
        fig.savefig("evaluation_statistics_plot_complexe.png")
    else:
        fig.savefig("evaluation_statistics_plot.png")

    plt.show()


if __name__ == "__main__":
    prompt()
    validate()
    statistics()
