import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
import os
import csv
import json


def search_available_devices():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

def statistics_to_csv(path:str, statistics):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for item in statistics:
            writer.writerow([item])

    
def initialize_weights_xavier(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

def count_correct_samples(pred : torch.tensor, labels : torch.tensor, threshold=0.5):
    return (((torch.sigmoid(pred) > threshold) == labels).all(dim=1)).sum().item()

def init_confusion(num_labels):
    confusion = {i : {
            'tp' : 0,
            'tn' : 0,
            'fp' : 0,
            'fn' : 0,
        } for i in range(num_labels)}
    return confusion

def accuracy(tp, tn, fp, fn):
    return (tp+tn)/(tp+tn+fp+fn)

def precision(tp, fp):
    if tp + fp == 0:
        return 0
    else:
        return tp/(tp+fp)

def recall(tp, fn):
    if tp + fn == 0:
        return 0
    else:
        return tp/(tp+fn)

def f1_score(tp, fp, fn):
    if (precision(tp, fp)+recall(tp, fn)) == 0:
        return 0
    else:
        return 2*((precision(tp, fp)*recall(tp, fn))/(precision(tp, fp)+recall(tp, fn)))
    
def class_weights(idx, task, data_path):

    if task == 'a':
        label_mapping = {
                'anger' : 0,
                'fear' : 1,
                'joy' : 2,
                'sadness' : 3, 
                'surprise' : 4, 
                'none' : 5
            }
    elif task == 'b':
        label_mapping = {
                'light anger' : 0, 
                'medium anger' : 1,
                'strong anger' : 2,
                'light fear' : 3,
                'medium fear' : 4, 
                'strong fear' : 5,
                'light joy' : 6, 
                'medium joy' : 7, 
                'strong joy' : 8,
                'light sadness' : 9, 
                'medium sadness' : 10, 
                'strong sadness' : 11,
                'light surprise' : 12,
                'medium surprise' : 13, 
                'strong surprise' : 14, 
                'none' : 15
        }

    # load json
    with open(os.path.join(os.path.dirname(__file__), f'../{data_path}')) as file:
        data = json.load(file)
    
    labels = []
    for entry in data:
        binary_labels = [0] * len(label_mapping)
        for label in entry['emotions']:
            if label in label_mapping:
                binary_labels[label_mapping[label]] = 1
        labels.append(binary_labels)
    labels = torch.tensor(labels)
    labels = labels[idx]
    
    class_counts = labels.sum(dim=0).float()
    print(class_counts)
    class_weights = 1.0 / (class_counts + 1e-6)
    print(class_weights)
    class_weights /= class_weights.sum()
    print(class_weights)
    sample_weights = (labels * class_weights).sum(dim=1)
    sample_weights = sample_weights.numpy()
    print(sample_weights)
    print(f'Length of sample weights: {len(sample_weights)}')
    return sample_weights

def clear_json():
    with open(os.path.join(os.path.dirname(__file__), f'../../data/merged_data.json')) as file:
        data = json.load(file)
    filtered_data = [entry for entry in data if entry.get("emotions") != ["disgust"]]
    with open(os.path.join(os.path.dirname(__file__), f'../../data/merged_data_c.json'), "w") as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)



if __name__=='__main__':
    clear_json()