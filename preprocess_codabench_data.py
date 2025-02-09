import csv
import json

from common import remove_junk

# Add True or False weather the emotions are complex or not
INPUT_PATH_1 = ("data/codabench_data/dev/eng_a.csv", False)
INPUT_PATH_2 = ("data/codabench_data/dev/eng_b.csv", True)
INPUT_PATH_3 = ("data/codabench_data/dev/eng_c.csv", False)


INPUT_PATHS = [INPUT_PATH_1, INPUT_PATH_2, INPUT_PATH_3]

PRINT_LINES = False


def main():
    for path, use_complex_emotions in INPUT_PATHS:
        # in path1 there are only binary emotions (either the emotion is there or not, 0 or 1)
        # in path2 there are complex for the bonus task (0 - 3)
        # this is important for the parsing below

        with open(path, "r") as f:
            data = csv.reader(f)

            dataset = []

            # Format: id,text,Anger,Fear,Joy,Sadness,Surprise
            # eng_train_track_a_00001,But not very happy.,0,0,1,1,0

            # skip first line
            skipped_first_line = False

            for line in data:
                assert len(line) == 7

                if not skipped_first_line:
                    skipped_first_line = True
                    continue

                # get the id
                id = line[0]

                # get the sentence
                sentence = remove_junk(line[1])

                emotions = []
                # get the emotions
                if use_complex_emotions:
                    if int(line[2]) == 1:
                        emotions.append("light anger")
                    if int(line[2]) == 2:
                        emotions.append("medium anger")
                    if int(line[2]) == 3:
                        emotions.append("strong anger")
                    if int(line[3]) == 1:
                        emotions.append("light fear")
                    if int(line[3]) == 2:
                        emotions.append("medium fear")
                    if int(line[3]) == 3:
                        emotions.append("strong fear")
                    if int(line[4]) == 1:
                        emotions.append("light joy")
                    if int(line[4]) == 2:
                        emotions.append("medium joy")
                    if int(line[4]) == 3:
                        emotions.append("strong joy")
                    if int(line[5]) == 1:
                        emotions.append("light sadness")
                    if int(line[5]) == 2:
                        emotions.append("medium sadness")
                    if int(line[5]) == 3:
                        emotions.append("strong sadness")
                    if int(line[6]) == 1:
                        emotions.append("light surprise")
                    if int(line[6]) == 2:
                        emotions.append("medium surprise")
                    if int(line[6]) == 3:
                        emotions.append("strong surprise")
                    if len(emotions) == 0:
                        emotions.append("none")
                else:
                    if int(line[2]) >= 1:
                        emotions.append("anger")
                    if int(line[3]) >= 1:
                        emotions.append("fear")
                    if int(line[4]) >= 1:
                        emotions.append("joy")
                    if int(line[5]) >= 1:
                        emotions.append("sadness")
                    if int(line[6]) >= 1:
                        emotions.append("surprise")
                    if len(emotions) == 0:
                        emotions.append("none")

                dataset.append({"id": id, "sentence": sentence, "emotions": emotions})

                if PRINT_LINES:
                    print("*" * 50)
                    print(
                        "Parsed line:",
                        {"id": id, "sentence": sentence, "emotions": emotions},
                    )
                    print("*" * 50)

            print("Dataset length:", len(dataset))

            # save the dataset
            with open(path.replace(".csv", "_parsed.json"), "w") as f:
                json.dump(dataset, f, indent=4)


if __name__ == "__main__":
    main()
