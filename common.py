import re


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


EMOTION_LABELS = [
    "anger",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "disgust",
    "none",
]


# => there is no disgust in the dataset
EMOTION_COMPLEX_LABELS = [
    "light anger",
    "medium anger",
    "strong anger",
    "light fear",
    "medium fear",
    "strong fear",
    "light joy",
    "medium joy",
    "strong joy",
    "light sadness",
    "medium sadness",
    "strong sadness",
    "light surprise",
    "medium surprise",
    "strong surprise",
    "none",
]


def get_precision(tp: int, fp: int) -> float:
    return round(tp / (tp + fp), 2)


def get_recall(tp: int, fn: int) -> float:
    return round(tp / (tp + fn), 2)


def get_f1_score(precision: int, recall: int) -> float:
    return round(2 * (precision * recall) / (precision + recall), 2)


def get_accuracy(tp: int, fp: int, tn: int, fn: int) -> float:
    return round((tp + tn) / (tp + fp + tn + fn), 2)


def get_f1_score_macro(labels, predicitions) -> float:
    assert len(labels) == len(predicitions)

    f1_scores = []

    for label, prediction in zip(labels, predicitions):
        solution = set(label)
        answer = set(prediction)

        total = len(solution.union(answer))

        true_negatives = 0
        for element in EMOTION_LABELS:
            if element not in solution and element not in answer:
                true_negatives += 1

        # get the intersection of the two sets and remove them from both sets
        intersection = answer.intersection(solution)
        true_positives = len(intersection)

        answer -= intersection
        solution -= intersection

        false_positives = len(answer)
        false_negatives = len(solution)

        assert true_positives + false_positives + false_negatives == total

        f1_scores.append(
            true_positives
            / (true_positives + 0.5 * (false_positives + false_negatives))
        )

    return round(sum(f1_scores) / len(f1_scores), 2)


def get_f1_score_weighted(labels, predicitions) -> float:
    assert len(labels) == len(predicitions)

    f1_scores = []
    support = []

    for label, prediction in zip(labels, predicitions):
        solution = set(label)
        answer = set(prediction)

        total = len(solution.union(answer))

        true_negatives = 0
        for element in EMOTION_LABELS:
            if element not in solution and element not in answer:
                true_negatives += 1

        # get the intersection of the two sets and remove them from both sets
        intersection = answer.intersection(solution)
        true_positives = len(intersection)

        answer -= intersection
        solution -= intersection

        false_positives = len(answer)
        false_negatives = len(solution)

        assert true_positives + false_positives + false_negatives == total

        f1_scores.append(
            true_positives
            / (true_positives + 0.5 * (false_positives + false_negatives))
        )

        support.append(true_positives + false_negatives)

    weighted = 0

    for i in range(len(labels)):
        weighted += support[i] * f1_scores[i]

    return round(weighted / sum(support), 2)
