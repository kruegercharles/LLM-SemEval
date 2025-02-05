import matplotlib.pyplot as plt
import json
import numpy as np
import os

def plot_acc():

    models = ['attention', 'deep', 'max', 'mean', 'pure']
    path_to_stats = os.path.join(os.path.dirname(__file__), f'../../outputs/statistics/task_a/small/')
    results = np.zeros(shape=(5,5,8))

    for i, model in enumerate(models):
        for j, stat in enumerate(os.listdir(os.path.join(path_to_stats, model))):
            with open(path_to_stats + "/" + model + "/" + stat, 'r') as file:
                data = json.load(file)
                results[i, j, :] = data['test']['f1']['micro']
    
    mean_accuracy = np.mean(results, axis=1) 
    std_accuracy = np.std(results, axis=1) 
    
    epochs = np.arange(1,9)
    colors = ['b', 'g', 'r', 'c', 'm']
    plt.figure(figsize=(10, 6)) 

    for i in range(5):
        plt.plot(epochs, mean_accuracy[i], marker='o', linestyle='-', color=colors[i], label=models[i])
        #plt.fill_between(epochs, mean_accuracy[i] - std_accuracy[i], mean_accuracy[i] + std_accuracy[i], color=colors[i], alpha=0.2)

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Model F1 Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__=="__main__":
    plot_acc()