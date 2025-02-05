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
from common import *  # noqa
import copy  # noqa

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

RUNS_TO_PROMT = 15


class ModelClass:
    def __init__(self, name: str, path: str):
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


class EnsembleVoting:
    def __init__(self, name: str):
        self.name = name

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


class Human:
    def __init__(self):
        self.name: str = "Human"
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


def get_human_feedback() -> list[str]:
    """
    This function is used to get human feedback for the predictions.
    """

    human_answer = []

    print("What emotions do you think are expressed in this sentence?")

    text = ""
    for emotion in EMOTION_LABELS:
        text += f"[{EMOTION_LABELS.index(emotion) + 1}] {emotion}, "
    print(text[:-2])
    print("[f] finish answer, [r] restart answer\n")

    while True:
        human_input = input()

        try:
            if human_input == "f":
                break
            if human_input == "r":
                human_answer.clear()
                continue

            emotion = EMOTION_LABELS[int(human_input) - 1]

            if emotion == "none" and len(human_answer) > 0:
                print(
                    "You can only select 'none' if no other emotion was selected, restarting"
                )
                human_answer.clear()
                continue
            if emotion == "none":
                human_answer.append("none")
                break

            # if answer already in list, continue
            if emotion in human_answer:
                continue

            human_answer.append(EMOTION_LABELS[int(human_input) - 1])

        except:  # noqa: E722
            continue

    if len(human_answer) == 0:
        human_answer.append("none")
    print("Human answer:", human_answer)
    return list(set(human_answer))


PROMPT_EXAMPLES: dict = load_dataset("data/codabench_data/dev/eng_a_parsed.json")

# Define threshold for binary classification
THRESHOLD = 0.5

models: list[ModelClass] = []

# models.append(
# ModelClass(name="RoBERTa base-model", path="models/roberta-base/")
# )  # base model

# models.append(
# ModelClass(name="Finetuned semeval", path="output/roberta-semeval/")
# )  # finetuned with codabench data
# models.append(
#     ModelClass(name="Finetuned emotions_data", path="output/emotions-data/")
# )  # finetuned with emotions data
# models.append(
#     ModelClass(name="Finetuned dair-ai", path="output/dair-ai/")
# )  # finetuned with dair-ai data
# models.append(
#     ModelClass(name="Finetuned goemotions", path="output/goemotions/")
# )  # finetuned with goemotions data
# models.append(
#     ModelClass(name="Finetuned merged_dataset", path="output/merged-dataset/")
# )  # finetuned with merged dataset


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


TOKENIZER_PATH = "models/roberta-base/"


def prompt():
    tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH, cache_dir="cache-dir/")

    name = ""

    print("\n\nPlease enter your name.")
    while name.strip() == "":
        # get name of user via input
        name = input()

    print("Hello", name, "!")

    human = Human()
    human.name = name

    overall = EnsembleVoting("Ensemble Voting")

    run = 1

    already_seen = []

    while run <= len(PROMPT_EXAMPLES) and run <= RUNS_TO_PROMT:
        # pick a random prompt
        prompt = random.choice(list(PROMPT_EXAMPLES.keys()))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # check if prompt was already seen
        if prompt in already_seen:
            continue
        already_seen.append(prompt)

        # get the solution
        solution = PROMPT_EXAMPLES[prompt]

        print("\n\n")
        print(run, '- Sentence: "' + prompt + '"')
        run += 1

        human_answer = get_human_feedback()

        for emotion in EMOTION_LABELS:
            if emotion in human_answer and emotion in solution:
                human.stats[emotion]["tp"] += 1
            elif emotion in human_answer and emotion not in solution:
                human.stats[emotion]["fp"] += 1
            elif emotion not in human_answer and emotion in solution:
                human.stats[emotion]["fn"] += 1
            elif emotion not in human_answer and emotion not in solution:
                human.stats[emotion]["tn"] += 1
            else:
                raise ValueError("Invalid state")

        voting_table = {}
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
            predicted_emotions = [
                EMOTION_LABELS[j] for j, val in enumerate(predicted_labels) if val == 1
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

            # print("Voting Table Models:", voting_table)

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
        )  # There must always be at least one emotion predicted
        assert ("none" not in final_answer) or (len(final_answer) == 1)

        final_answer_text = "Answer Modells:"

        if not final_answer:
            print(final_answer_text, "none")
        else:
            print(final_answer_text, final_answer)

        for emotion in EMOTION_LABELS:
            if emotion in final_answer and emotion in solution:
                overall.stats[emotion]["tp"] += 1
            elif emotion in final_answer and emotion not in solution:
                overall.stats[emotion]["fp"] += 1
            elif emotion not in final_answer and emotion in solution:
                overall.stats[emotion]["fn"] += 1
            elif emotion not in final_answer and emotion not in solution:
                overall.stats[emotion]["tn"] += 1
            else:
                raise ValueError("Invalid state")

        print("Correct answer:", solution)

    return human, overall


def validate(human: Human, overall: EnsembleVoting):
    # check if the values makes sence
    total = RUNS_TO_PROMT
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

    for emotion in EMOTION_LABELS:
        sum = (
            human.stats[emotion]["tp"]
            + human.stats[emotion]["fp"]
            + human.stats[emotion]["tn"]
            + human.stats[emotion]["fn"]
        )
        if sum != total:
            print("Model: ", human.name)
            print("State: ", human.stats)
            print("Emotion: ", emotion)
            print("Total: ", total)
            print("Sum: ", sum)
            raise ValueError("Invalid state")

        sum = (
            overall.stats[emotion]["tp"]
            + overall.stats[emotion]["fp"]
            + overall.stats[emotion]["tn"]
            + overall.stats[emotion]["fn"]
        )
        if sum != total:
            print("Model: ", overall.name)
            print("State: ", overall.stats)
            print("Emotion: ", emotion)
            print("Total: ", total)
            print("Sum: ", sum)
            raise ValueError("Invalid state")


def statistics(human: Human, overall: EnsembleVoting):
    output_data: list[str] = []

    # Calculate statistics
    print(" ")
    output_data.append("Statistics:")

    print(human.name, human.stats)

    output_data.append(f"Human '{human.name}':")

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
        tp = human.stats[emotion]["tp"]
        tps += tp
        fp = human.stats[emotion]["fp"]
        fps += fp
        tn = human.stats[emotion]["tn"]
        tns += tn
        fn = human.stats[emotion]["fn"]
        fns += fn

        f1 = get_f1_score(tp=tp, fp=fp, fn=fn)

        f1_scores_macro.append(f1)
        accuracies.append(get_accuracy(tp=tp, fp=fp, tn=tn, fn=fn))
        precisions.append(get_precision(tp=tp, fp=fp))
        recalls.append(get_recall(tp=tp, fn=fn))

        support.append(tp + fn)
        f1_scores_weighted.append(f1 * (tp + fn))

    human.precision = round(sum(precisions) / len(precisions), 2)
    human.recall = round(sum(recalls) / len(recalls), 2)
    human.accuracy = round(sum(accuracies) / len(accuracies), 2)
    human.f1_score_micro = round(get_f1_score(tp=tps, fp=fps, fn=fns), 2)
    human.f1_score_macro = round(sum(f1_scores_macro) / len(f1_scores_macro), 2)
    human.f1_score_weighted = round(sum(f1_scores_weighted) / sum(support), 2)

    output_data.append("Human Precision: " + str(human.precision))
    output_data.append("Human Recall: " + str(human.recall))
    output_data.append("Human Accuracy: " + str(human.accuracy))
    output_data.append("Human F1-Score micro: " + str(human.f1_score_micro))
    output_data.append("Human F1-Score macro: " + str(human.f1_score_macro))
    output_data.append("Human F1-Score weighted: " + str(human.f1_score_weighted))
    output_data.append(" ")

    output_data.append(f"Overall '{overall.name}':")

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
        tp = overall.stats[emotion]["tp"]
        tps += tp
        fp = overall.stats[emotion]["fp"]
        fps += fp
        tn = overall.stats[emotion]["tn"]
        tns += tn
        fn = overall.stats[emotion]["fn"]
        fns += fn

        f1 = get_f1_score(tp=tp, fp=fp, fn=fn)

        f1_scores_macro.append(f1)
        accuracies.append(get_accuracy(tp=tp, fp=fp, tn=tn, fn=fn))
        precisions.append(get_precision(tp=tp, fp=fp))
        recalls.append(get_recall(tp=tp, fn=fn))

        support.append(tp + fn)
        f1_scores_weighted.append(f1 * (tp + fn))

    overall.precision = round(sum(precisions) / len(precisions), 2)
    overall.recall = round(sum(recalls) / len(recalls), 2)
    overall.accuracy = round(sum(accuracies) / len(accuracies), 2)
    overall.f1_score_micro = round(get_f1_score(tp=tps, fp=fps, fn=fns), 2)
    overall.f1_score_macro = round(sum(f1_scores_macro) / len(f1_scores_macro), 2)
    overall.f1_score_weighted = round(sum(f1_scores_weighted) / sum(support), 2)

    output_data.append("Overall Precision: " + str(overall.precision))
    output_data.append("Overall Recall: " + str(overall.recall))
    output_data.append("Overall Accuracy: " + str(overall.accuracy))
    output_data.append("Overall F1-Score micro: " + str(overall.f1_score_micro))
    output_data.append("Overall F1-Score macro: " + str(overall.f1_score_macro))
    output_data.append("Overall F1-Score weighted: " + str(overall.f1_score_weighted))
    output_data.append(" ")

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
        output_data.append("Accuracy: " + str(model.accuracy))
        output_data.append("F1-Score micro: " + str(model.f1_score_micro))
        output_data.append("F1-Score macro: " + str(model.f1_score_macro))
        output_data.append("F1-Score weighted: " + str(model.f1_score_weighted))
        output_data.append(" ")
        output_data.append("*" * 50)
        output_data.append(" ")

    # Print statistics
    for line in output_data:
        print(line)

    new_data = []
    for line in output_data:
        new_data.append(line + "\n")

    # Read current statistics from file
    with open("human-evaluation.txt", "r") as f:
        old_data = f.readlines()

    all_data = old_data + new_data

    # Save statistics to file
    with open("human-evaluation.txt", "w") as f:
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
    human, overall = prompt()
    validate(human=human, overall=overall)
    statistics(human=human, overall=overall)
