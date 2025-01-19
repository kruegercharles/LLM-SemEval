import json
import random

import pandas as pd
from common import EMOTION_LABELS, remove_junk

# DATASET: https://huggingface.co/datasets/dair-ai/emotion
INPUT_PATH_1 = "data/dair-ai/train-00000-of-00001.parquet"
paths = [INPUT_PATH_1]

emotions_amount_output = {}
for emotion in EMOTION_LABELS:
    emotions_amount_output[emotion] = 0

emotions_amount_input = {}


def map_emotion(label: int) -> str:
    if label == 0:
        return "sadness"
    if label == 1:
        return "joy"
    if label == 2:
        return "love"
    if label == 3:
        return "anger"
    if label == 4:
        return "fear"
    if label == 5:
        return "surprise"

    raise ValueError("Unknown label:", label)


def main():
    all_data = []

    id = 1

    for path in paths:
        print("Start with path:", path)
        data = pd.read_parquet(path)

        # There are 4 columns: id, text, label, label_text
        # Iterate over the rows and find every different label_text
        emotions = set()
        for _, row in data.iterrows():
            emotion = map_emotion(row["label"])

            emotions.add(emotion)
            emotions_amount_input[emotion] = emotions_amount_input.get(emotion, 0) + 1
        print("Emotions:", emotions)

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

            emotion_found = map_emotion([row["label"]][0])

            if emotion_found not in emotions:
                print("Emotion not found:", emotion_found)
                print('[row["label"]]:', [row["label"]])
                print('[row["label"]][0]:', [row["label"]][0])
                raise ValueError("Emotion not found")

            if emotion_found in EMOTION_LABELS:
                emotion = emotion_found
                emotions_amount_output[emotion] += 1
            elif emotion_found == "neutral":
                emotion = "none"
                emotions_amount_output["none"] += 1
            # elif (
            #     emotion_found == "love"
            #     or emotion_found == "amusement"
            #     or emotion_found == "relief"
            # ):emotions
            #     emotion = ["joy"]
            #     emotions_amount_output["joy"] += 1
            # elif emotion_found == "grief":
            #     emotion = ["sadness"]
            #     emotions_amount_output["sadness"] += 1
            else:
                continue

            data = {
                "id": id,
                "sentence": remove_junk(row["text"]),
                "emotions": [emotion],
            }
            all_data.append(data)
            id += 1

    print("\nLength of all data:", len(all_data), "\n")

    print("\nEmotions amount input:", emotions_amount_input)
    print("Emotions amount output:", emotions_amount_output)

    # remove all keys with value 0 from emotions_amount_output
    for key in list(emotions_amount_output.keys()):
        if emotions_amount_output[key] == 0:
            del emotions_amount_output[key]

    # find the key in emotions_amount_output with the smallest value
    min_value = min(emotions_amount_output.values())
    # save the associated key
    min_key = [
        key
        for key in emotions_amount_output
        if emotions_amount_output[key] == min_value
    ][0]

    emotions_for_pruning = {
        "anger": 0,
        "fear": 0,
        "joy": 0,
        "sadness": 0,
        "surprise": 0,
        "disgust": 0,
        "none": 0,
    }
    # go through all values and keep from every emotion maximum 4 times the min-value. Pick randomly
    random.shuffle(all_data)

    length_data = len(all_data)
    index = 1

    pruned_data = []

    print("Shuffled all data")
    for item in all_data:
        if index % 10000 == 0:
            print(
                "Processed",
                index,
                "/",
                length_data,
                "rows =>",
                round(index / length_data * 100, 2),
                "%",
            )
        index += 1

        # if item["emotions"] is not list:
        # print("item:", item)
        # print("item[emotions]:", item["emotions"])
        # print("type(item[emotions]):", type(item["emotions"]))
        # raise ValueError("item[emotions] is not a list")

        emotion = item["emotions"][0]
        # print("Emotion:", emotion)
        if emotion not in emotions_for_pruning.keys():
            print("Emotion not in emotions_for_pruning.keys():", emotion)
        assert emotion in emotions_for_pruning.keys()
        if emotions_for_pruning.get(emotion) < (min_value * 4):
            emotions_for_pruning[emotion] += 1
            pruned_data.append(item)

    print("Min value:", min_value, "for key:", min_key)
    print("Length of all data after pruning:", len(pruned_data))
    print("Emotions for pruning:", emotions_for_pruning)

    # Save the data to a json file
    with open("data/dair-ai/parsed_data.json", "w", encoding="utf-8") as f:
        json.dump(pruned_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
