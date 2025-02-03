import json

from common import EMOTION_LABELS

PATH_1 = "data/codabench_data/train/eng_a_parsed.json"
PATH_2 = "data/dair-ai/parsed_data.json"
# PATH_3 = "data/Emotions_Data/parsed_data.json"
PATH_3 = "data/Emotions_Data/parsed_data_unpruned.json"
PATH_4 = "data/go_emotions/parsed_data.json"
paths = []
paths.append(PATH_1)
paths.append(PATH_2)
paths.append(PATH_3)
paths.append(PATH_4)

merged_data = []
id = 1

emotion_counter = {}

for emotion in EMOTION_LABELS:
    emotion_counter[emotion] = 0

sum = 0

for path in paths:
    with open(path) as f:
        data = json.load(f)

    sum += len(data)

    for entry in data:
        if id % 5000 == 0:
            print("Merged", id, "entries out of", sum)

        entry["id"] = id
        id += 1
        merged_data.append(entry)
        for emotion in entry["emotions"]:
            emotion_counter[emotion] += 1

print("Emotion Counter:", emotion_counter)
print("Total:", id)

with open("data/merged_data.json", "w") as f:
    json.dump(merged_data, f, indent=4)
