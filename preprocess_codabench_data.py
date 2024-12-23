import json
import copy

INPUT_PATH_1 = "data/codabench_data/train/eng_a.csv"
INPUT_PATH_2 = "data/codabench_data/train/eng_b.csv"


INPUT_PATHS = [INPUT_PATH_1, INPUT_PATH_2]


def parse_line(line: str) -> list:
    print(" ")
    print("*" * 50)
    print("Input line:", line, end="")

    # the lines are completely messed up, so we have to split them manually
    # sometimes they contain quotes, sometimes they don't
    # sometimes they contain commas within the quote, so we can't just split by comma
    # sometimes they contain 3 quotes in a row
    # sometimes they contain quotes within quotes

    # remove all quotes, single or double
    line = line.replace("'", "")
    line = line.replace('"', "")

    # remove \n
    line = line.replace("\n", "")

    # remove ~
    line = line.replace("~", "")

    # remove `
    line = line.replace("`", "")

    # remove < or >
    line = line.replace("<", "")
    line = line.replace(">", "")

    # remove ;
    line = line.replace(";", "")

    # remove &
    line = line.replace("&", "")

    # remove #
    line = line.replace("#", "")

    # remove ;
    line = line.replace(";", "")

    # remove "nbsp;"
    line = line.replace("nbsp;", "")

    # remove *
    line = line.replace("*", "")

    # remove "..."
    line = line.replace("...", "")

    # remove spaces at the beginning and end
    line = line.strip()

    # State variables
    parts = []

    # the first characters until the first comma are always the id
    id = copy.deepcopy(line).split(",", 1)[0]
    parts.append(id)

    # remove the id from the line
    line = line.replace((id + ","), "")

    # reverse the line
    line = line[::-1]

    # go through all the emotions
    # usually there is a 0 or 1, but sometimes there is just no value, which means 0
    num_found = 0
    emotions = []

    # create a copy so we can always remove every letter from the original line
    line_copy = copy.deepcopy(line)

    for letter in line_copy:
        line = line[1:]
        if letter == ",":
            emotions.append(num_found)
            num_found = 0
        elif letter.isdigit():
            num_found = int(letter)
        else:
            raise ValueError(f"Unexpected letter: {letter}")

        # there are 5 emotions
        if len(emotions) == 5:
            break

    # reverse emotions to get the correct order
    emotions.reverse()

    if len(emotions) != 5:
        print("Line:", line)
        print("Emotions:", emotions)
        raise ValueError(f"Expected 5 emotions but got {len(emotions)}. Line: {line}")

    # reverse line to get the correct order
    line = line[::-1]

    # remove double whitespace within the line
    line = line.replace("  ", " ")

    # remove whitespace at the beginning and end
    line = line.strip()

    # the sentence is what is left of the line
    parts.append(line)

    # add the emotions to the parts
    parts.extend(emotions)

    # Construct result dictionary
    if len(parts) != 7:
        print("Line:", line)
        print("Parts:", parts)
        raise ValueError(f"Expected 7 parts but got {len(parts)}. Line: {line}")

    print("Line Parse Result:", parts)
    print("*" * 50)

    return parts


def main():
    for path in INPUT_PATHS:
        with open(path, "r") as f:
            data = f.readlines()

        print("Current file:", path)

        dataset = []

        # Format: id,text,Anger,Fear,Joy,Sadness,Surprise
        # eng_train_track_a_00001,But not very happy.,0,0,1,1,0

        # skip first line
        skipped_first_line = False

        # split line at comma
        for line in data:
            if not skipped_first_line:
                skipped_first_line = True
                continue

            line_parsed = parse_line(line)
            assert len(line_parsed) == 7

            # get the id
            id = line_parsed[0]

            # get the sentence
            sentence = line_parsed[1]

            emotions = []
            # get the emotions
            if int(line_parsed[2]) >= 1:
                emotions.append("Anger")
            if int(line_parsed[3]) >= 1:
                emotions.append("Fear")
            if int(line_parsed[4]) >= 1:
                emotions.append("Joy")
            if int(line_parsed[5]) >= 1:
                emotions.append("Sadness")
            if int(line_parsed[6]) >= 1:
                emotions.append("Surprise")
            if len(emotions) == 0:
                emotions.append("None")

            dataset.append({"id": id, "sentence": sentence, "emotions": emotions})

        print("Dataset length:", len(dataset))

        # save the dataset
        with open(path.replace(".csv", "_parsed.json"), "w") as f:
            json.dump(dataset, f, indent=4)


if __name__ == "__main__":
    main()
