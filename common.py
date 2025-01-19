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
