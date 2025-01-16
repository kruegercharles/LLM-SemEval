import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
import csv


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