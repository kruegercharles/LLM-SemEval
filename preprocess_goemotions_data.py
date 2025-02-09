import json

import pandas as pd
from common import EMOTION_LABELS, remove_junk

# DATASET: https://huggingface.co/datasets/google-research-datasets/go_emotions
INPUT_PATH_1 = "data/go_emotions/train-00000-of-00001.parquet"
paths = [INPUT_PATH_1]

emotion_distribution = {}

for emotion in EMOTION_LABELS:
    emotion_distribution[emotion] = 0


def get_emotions(row) -> list:
    #     EMOTION_LABELS = [
    #     "anger",
    #     "fear",
    #     "joy",
    #     "sadness",
    #     "surprise",
    #     "disgust",
    #     "none",
    # ]

    emotions = []

    if row["anger"] == 1:
        emotions.append("anger")
        emotion_distribution["anger"] += 1

    if row["disgust"] == 1:
        emotions.append("disgust")
        emotion_distribution["disgust"] += 1

    if row["fear"] == 1:
        emotions.append("fear")
        emotion_distribution["fear"] += 1

    if row["joy"] == 1:
        emotions.append("joy")
        emotion_distribution["joy"] += 1

    if row["sadness"] == 1:
        emotions.append("sadness")
        emotion_distribution["sadness"] += 1

    if row["surprise"] == 1:
        emotions.append("surprise")
        emotion_distribution["surprise"] += 1

    if row["neutral"] == 1:
        emotions.append("none")
        emotion_distribution["none"] += 1

    return emotions


def main():
    all_data = []

    id = 1

    for path in paths:
        print("Start with path:", path)
        data = pd.read_parquet(path)

        rows = len(data)

        print("Original length of test data:", rows)

        # Go through every row and create a dictionary
        for index, row in data.iterrows():
            if index % 10000 == 0:
                print(
                    "Processed",
                    index,
                    "/",
                    rows,
                    "rows =>",
                    round(index / rows * 100, 2),
                    "%",
                )

            if row["example_very_unclear"] == "true":
                print("Example very unclear:", row["text"])
                continue

            emotions = get_emotions(row)
            if len(emotions) == 0:
                continue

            data = {
                "id": id,
                "sentence": remove_junk(row["text"]),
                "emotions": emotions,
            }
            all_data.append(data)
            id += 1

    print("\nLength of all data:", len(all_data), "\n")

    print("Emotion distribution:", emotion_distribution)

    # Save the data to a json file
    with open("data/go_emotions/parsed_data.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
