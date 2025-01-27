import torch
import hydra
import json
import numpy as np
import os
import torch.utils
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import torch.utils.data
from transformers import RobertaTokenizer
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from misc.misc import search_available_devices, plot_losses, statistics_to_csv, count_correct_samples, initialize_weights_xavier
from data.dataset import EmotionData
from models.lm_classifier import *
from sklearn.metrics import f1_score
from torchviz import make_dot

def select_model(name, backbone, num_labels, device):
    if name == 'base':
        return RobertaForSequenceClassificationPure(backbone, num_labels).to(device=device)
    elif name == 'deep':
        return RobertaForSequenceClassificationDeep(backbone, num_labels).to(device=device)
    elif name == 'mean':
        return RobertaForSequenceClassificationMeanPooling(backbone, num_labels).to(device=device)
    elif name == 'max':
        return RobertaForSequenceClassificationMaxPooling(backbone, num_labels).to(device=device)
    elif name == 'attention':
        return RobertaForSequenceClassificationAttentionPooling(backbone, num_labels).to(device=device)
    else:
        raise ValueError('Specified model name is not available!')


def train(fold, epochs, model, train_loader, val_loader, test_loader, optimizer, criterion, device):

    train_losses, val_losses, train_acc, val_acc, f1_train, f1_val = list(), list(), list(), list(), list(), list()

    for epoch in range(epochs):
        print(f'Start Training Epoch {epoch+1}.')
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        f1_t = []
        for batch in train_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            f1 = f1_score(labels.detach().cpu().numpy(), (torch.where(torch.sigmoid(outputs) >=0.5, torch.tensor(1.0), torch.tensor(0.0))).detach().cpu().numpy(), average='macro', zero_division=0.0)
            f1_t.append(f1)
            correct_train += count_correct_samples(outputs, labels) / len(batch['input_ids'])
            total_train_loss += loss.item()

        train_losses.append(total_train_loss)
        train_acc.append(correct_train / len(train_loader))
        f1_train.append(sum(f1_t) / len(f1_t))

        # save model checkpoint:
        checkpoint = {
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss' : total_train_loss
        }

        torch.save(checkpoint, os.path.join(os.path.dirname(__file__), f'../outputs/models/{model.name}_fold_{fold}_epoch_{epoch+1}.pth'))

        # Validation step
        model.eval()
        total_val_loss = 0
        correct_val = 0
        f1_v = []
        with torch.no_grad():
            print(f'Start Validation Epoch {epoch+1}.')
            for batch in val_loader:
                input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                #print(outputs.logits)
                #print(f'Inputs: {labels}')
                #print(f'Outputs: {outputs}')
                loss = criterion(outputs, labels.float())
                f1 = f1_score(labels.detach().cpu().numpy(), (torch.where(torch.sigmoid(outputs) >=0.5, torch.tensor(1.0), torch.tensor(0.0))).detach().cpu().numpy(), average='macro', zero_division=0.0)
                f1_v.append(f1)
                correct_val += (count_correct_samples(outputs, labels) / len(batch['input_ids']))
                total_val_loss += loss.item()

        print("Predictions: ", torch.where(torch.sigmoid(outputs) >=0.5, torch.tensor(1.0), torch.tensor(0.0)))
        print("Labels: ", labels)
        val_losses.append(total_val_loss)
        val_acc.append(correct_val / len(val_loader))
        f1_val.append(sum(f1_v) / len(f1_v))
        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Validation Loss = {val_losses[-1]:.4f}, Validation Acc. = {val_acc[-1]:.4f}")
    
    ### Test Model on SemEval Test Dataset ###
    model.eval()
    with torch.no_grad():
        print(f'Start Testing on Test Dataset!')
        for batch in test_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels.float())


    statistics_to_csv(os.path.join(os.path.dirname(__file__), f'../outputs/statistics/train_loss_fold_{fold}.csv'), train_losses)
    statistics_to_csv(os.path.join(os.path.dirname(__file__), f'../outputs/statistics/validation_loss_fold_{fold}.csv'), val_losses)
    statistics_to_csv(os.path.join(os.path.dirname(__file__), f'../outputs/statistics/train_acc_fold_{fold}.csv'), train_acc)
    statistics_to_csv(os.path.join(os.path.dirname(__file__), f'../outputs/statistics/validation_acc_fold_{fold}.csv'), val_acc)
    statistics_to_csv(os.path.join(os.path.dirname(__file__), f'../outputs/statistics/train_f1_fold_{fold}.csv'), f1_train)
    statistics_to_csv(os.path.join(os.path.dirname(__file__), f'../outputs/statistics/validation_f1_fold_{fold}.csv'), f1_val)

@hydra.main(config_path='../configs', config_name='train_roberta')
def cross_validation(cfg: DictConfig):
    #search for available devices
    device = search_available_devices()
    print(f'*** Found the following device: {device}. Start configuting training. ***')

    #dataset = EmotionData(os.path.join(os.path.dirname(__file__), cfg.data), cfg.backbone)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    dataset = EmotionData(os.path.join(os.path.dirname(__file__), cfg.data), cfg.backbone)
    test_dataset = EmotionData(os.path.join(os.path.dirname(__file__), '../data/eng_a_parsed_test.json'), cfg.backbone)
    
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dataset)))):
        print(f'Starting fold {fold+1}/{5}.')

        # create train and validation datasets
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)

        # create dataloaders 
        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

        model = select_model(cfg.model_name, cfg.backbone, cfg.num_labels, device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.1)
        criterion = nn.BCEWithLogitsLoss().to(device=device)

        train(fold+1, cfg.epochs, model, train_loader, val_loader, test_loader, optimizer, criterion, device)

if __name__ == '__main__':
    cross_validation()
