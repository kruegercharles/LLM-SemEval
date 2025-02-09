import json
import random

from common import EMOTION_LABELS

IMPORT_PATH = "data/merged_data.json"
EXPORT_PATH = "data/data_15k.json"

MAX_ENTRIES = 15000
MAX_ENTRIES_PER_EMOTION = int(round(MAX_ENTRIES / len(EMOTION_LABELS), None))
print("Max entries per emotion:", MAX_ENTRIES_PER_EMOTION)

# set random seed
random.seed(42)

emotion_counter_input = {}
emotion_counter_output = {}
for emotion in EMOTION_LABELS:
    emotion_counter_input[emotion] = 0
    emotion_counter_output[emotion] = 0

newData = []
with open(IMPORT_PATH, "r") as f:
    data = json.load(f)

# Count all emotions
for entry in data:
    emotions = entry["emotions"]
    for emotion in emotions:
        if emotion not in EMOTION_LABELS:
            print("Error: Unknown emotion", emotion)
            continue
        emotion_counter_input[emotion] += 1

print("Input data:", emotion_counter_input)

# Shuffle the data
random.shuffle(data)

print("Length of input data:", len(data))

known_sentences = set()


# Create a new dataset with a maximum of MAX_ENTRIES_PER_EMOTION per emotion
for entry in data:
    emotions = entry["emotions"]

    can_be_added = True

    for emotion in emotions:
        if emotion not in EMOTION_LABELS:
            print("Error: Unknown emotion", emotion)
            continue
        # Check if one of the emotions is already at the maximum
        if emotion_counter_output[emotion] > MAX_ENTRIES_PER_EMOTION:
            can_be_added = False
            break

    if can_be_added:
        # Check if the sentence is already in the dataset
        if entry["sentence"] in known_sentences:
            continue
        known_sentences.add(entry["sentence"])
        for emotion in emotions:
            emotion_counter_output[emotion] += 1

        newData.append(entry)

# Shuffle the data again
random.shuffle(newData)

print("Output data:", emotion_counter_output)
print("Length of new data:", len(newData))

with open(EXPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(newData, f, indent=4, ensure_ascii=False)
