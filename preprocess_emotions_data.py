import json
import re

import pandas as pd

# DATASET: https://huggingface.co/datasets/Villian7/Emotions_Data

INPUT_PATH_1 = "data/Emotions_Data/test-00000-of-00001-a6072f77b7d062e8.parquet"
INPUT_PATH_2 = "data/Emotions_Data/train-00000-of-00001-bd90147cb7ed1119.parquet"
INPUT_PATH_3 = "data/Emotions_Data/validation-00000-of-00001-51e4f3beb5529153.parquet"
paths = [INPUT_PATH_1, INPUT_PATH_2, INPUT_PATH_3]

EMOTION_LABELS = [
    "anger",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "disgust",
    "none",
]

emotions_amount_output = {
    "anger": 0,
    "fear": 0,
    "joy": 0,
    "sadness": 0,
    "surprise": 0,
    "disgust": 0,
    "none": 0,
}

emotions_amount_input = {}


def remove_junk(line: str) -> str:
    line = line.replace("\n", "")
    line = line.replace("~", "")
    line = line.replace("`", "")
    line = line.replace("<", "")
    line = line.replace(">", "")
    line = line.replace(";", "")
    line = line.replace("&", "")
    line = line.replace("#", "")
    line = line.replace("nbsp;", "")
    line = line.replace("*", "")
    line = line.replace("''", "")
    line = line.replace("{", "")
    line = line.replace("}", "")
    line = line.replace("[NAME]", "")

    while line.count("  ") > 0:
        line = line.replace("  ", " ")

    # remove urls
    while re.search(r"http\S+", line):
        line = re.sub(r"http\S+", "", line)

    while re.search(r"www\.\[a-zA-Z0-9]+", line):
        line = re.sub(r"www\.\[a-zA-Z0-9]+", "", line)

    # remove all words that start with @
    while re.search(r"@\S+", line):
        line = re.sub(r"@\S+", "", line)

    # remove still like \ud83d or \ude0d or similar
    # while re.search(r"\\u\S+", line):
    # line = re.sub(r"\\u\S+", "", line)

    # remove all words that start with \
    while re.search(r"\\\S+", line):
        line = re.sub(r"\\\S+", "", line)

    # remove \r
    while re.search(r"\\r", line):
        line = re.sub(r"\\r", "", line)

    # if there is only one " in the line, remove it
    if line.count('"') == 1:
        line = line.replace('"', "")

    if line.endswith(":"):
        line = line[:-1]

    # remove spaces at the beginning and end
    line = line.strip()

    return line


def main():
    all_data = []

    id = 1

    for path in paths:
        print("Start with path:", path)
        data = pd.read_parquet(path)

        # There are 4 columns: id, text, label, label_text
        # Iterate over the rows and find every different label_text
        # emotions = set()
        for _, row in data.iterrows():
            # emotions.add(row["label_text"])
            emotions_amount_input[row["label_text"]] = (
                emotions_amount_input.get(row["label_text"], 0) + 1
            )

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

            emotion_found = [row["label_text"]][0]
            if emotion_found in EMOTION_LABELS:
                emotion = emotion_found
                emotions_amount_output[emotion] += 1
            elif emotion_found == ["neutral"]:
                emotion = ["none"]
                emotions_amount_output["none"] += 1
            elif (
                emotion_found == ["love"]
                or emotion_found == ["amusement"]
                or emotion_found == ["relief"]
            ):
                emotion = ["joy"]
                emotions_amount_output["joy"] += 1
            elif emotion_found == ["grief"]:
                emotion = ["sadness"]
                emotions_amount_output["sadness"] += 1
            else:
                continue

            data = {
                "id": id,
                "sentence": remove_junk(row["text"]),
                "emotions": emotion,
            }
            all_data.append(data)
            id += 1

    print("\nLength of all data:", len(all_data), "\n")

    print("\nEmotions amount input:", emotions_amount_input, "\n")
    print("\nEmotions amount output:", emotions_amount_output, "\n")

    # Save the data to a json file
    with open("data/Emotions_Data/parsed_data.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
