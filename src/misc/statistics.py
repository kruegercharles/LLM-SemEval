import json 
import os


def calculate_dset_stats():
    
    path = os.path.join(os.path.dirname(__file__), f'../../data/eng_a_parsed_test.json')

    with open(path, 'r') as file:
        data = json.load(file)

    number_of_samples = len(data)
    print(f"Total number of data samples: {number_of_samples}")
    anger = 0
    fear = 0
    joy = 0
    sadness = 0
    surprise = 0
    none = 0
    total_labels = 0

    one = 0
    two = 0
    three = 0
    four = 0 
    five = 0 
    for sample in data:
        length = len(sample["emotions"])
        if length == 1:
            one += 1
        elif length == 2:
            two += 1
        elif length == 3: 
            three += 1
        elif length == 4:
            four += 1
        elif length == 5:
            five +=1
        for label in sample["emotions"]:
            total_labels += 1
            if label == "anger":
                anger += 1
            elif label == "fear":
                fear += 1
            elif label == "joy":
                joy += 1
            elif label == "sadness":
                sadness += 1
            elif label == "surprise":
                surprise += 1
            elif label == "none":
                none += 1
    print(f"Number anger total: {anger} , in percent: {anger/total_labels}")
    print(f"Number fear total: {fear} , in percent: {fear/total_labels}")
    print(f"Number joy total: {joy} , in percent: {joy/total_labels}")
    print(f"Number sadness total: {sadness} , in percent: {sadness/total_labels}")
    print(f"Number surprise total: {surprise} , in percent: {surprise/total_labels}")
    print(f"Number none total: {none} , in percent: {none/total_labels}")

    print(f"Number of samples with one emotion or none: {one} , in percent: {one/number_of_samples}")
    print(f"Number of samples with two emotions: {two} , in percent: {two/number_of_samples}")
    print(f"Number of samples with three emotions: {three} , in percent: {three/number_of_samples}")
    print(f"Number of samples with four emotions: {four} , in percent: {four/number_of_samples}")
    print(f"Number of samples with five emotions: {five} , in percent: {five/number_of_samples}")


calculate_dset_stats()