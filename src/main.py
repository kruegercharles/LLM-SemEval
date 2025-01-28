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
from misc.misc import search_available_devices, statistics_to_csv, count_correct_samples, init_confusion, accuracy, precision, recall, f1_score
from data.dataset import EmotionData
from models.lm_classifier import *

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


def train(fold, epochs, num_labels, model, train_loader, val_loader, test_loader, optimizer, criterion, device):

    train_losses, val_losses, test_losses = list(), list(), list()

    stats = {
        'train' : {
            'loss': list(),
            'accuracy' : {
                'macro' : list(),
                'micro' : list(),
            },
            'precision' : {
                'macro' : list(),
                'micro' : list(),
            },
            'recall' : {
                'macro' : list(), 
                'micro' : list(),
            },
            'f1' : {
                'macro' : list(), 
                'micro' : list(),
            }
        },
        'val' : {
            'loss': list(),
            'accuracy' : {
                'macro' : list(),
                'micro' : list(),
            },
            'precision' : {
                'macro' : list(),
                'micro' : list(),
            },
            'recall' : {
                'macro' : list(), 
                'micro' : list(),
            },
            'f1' : {
                'macro' : list(), 
                'micro' : list(),
            }
        },
        'test' : {
            'loss': list(),
            'accuracy' : {
                'macro' : list(),
                'micro' : list(),
            },
            'precision' : {
                'macro' : list(),
                'micro' : list(),
            },
            'recall' : {
                'macro' : list(), 
                'micro' : list(),
            },
            'f1' : {
                'macro' : list(), 
                'micro' : list(),
            }
        }
    }

    confusion = init_confusion()

    for epoch in range(epochs):
        print(f'Start Training Epoch {epoch+1}.')
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # convert output to binary format
            bin_out = torch.where(torch.sigmoid(outputs) >=0.5, torch.tensor(1.0), torch.tensor(0.0))
            # collect confusion statistics:
            for i in range(num_labels):
                for j in range(len(bin_out)):
                    if bin_out[j][i] == 1.0 and labels[j][i] == 1.0:
                        confusion[i]['tp'] += 1
                    elif bin_out[j][i] == 0.0 and labels[j][i] == 0.0:
                        confusion[i]['tn'] += 1
                    elif bin_out[j][i] == 1.0 and labels[j][i] == 0.0:
                        confusion[i]['fp'] += 1
                    elif bin_out[j][i] == 0.0 and labels[j][i] == 1.0:
                        confusion[i]['fn'] += 1

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_losses.append(total_train_loss)
        # calculate confusion stats in macro and micro strategy manner
        # macro strategy
        acc = [accuracy(confusion[i]['tp'], confusion[i]['tn'], confusion[i]['fp'], confusion[i]['fn']) for i in range(num_labels)]
        prec = [precision(confusion[i]['tp'], confusion[i]['fp']) for i in range(num_labels)]
        rec = [recall(confusion[i]['tp'], confusion[i]['fn']) for i in range(num_labels)]
        f1 = [f1_score(confusion[i]['tp'], confusion[i]['fp'], confusion[i]['fn']) for i in range(num_labels)]
        # insert values in dict
        stats['train']['accuracy']['macro'].append(sum(acc)/len(acc))
        stats['train']['precision']['macro'].append(sum(prec)/len(prec))
        stats['train']['recall']['macro'].append(sum(rec)/len(rec))
        stats['train']['f1']['macro'].append(sum(f1)/len(f1))
        # micro strategy
        tps = sum([confusion[i]['tp'] for i in range(num_labels)])
        tns = sum([confusion[i]['tn'] for i in range(num_labels)])
        fps = sum([confusion[i]['fp'] for i in range(num_labels)])
        fns = sum([confusion[i]['fn'] for i in range(num_labels)])
        # insert values in dict
        stats['train']['accuracy']['micro'].append(accuracy(tps, tns, fps, fns))
        stats['train']['precision']['micro'].append(precision(tps, fps))
        stats['train']['recall']['micro'].append(recall(tps, fns))
        stats['train']['f1']['micro'].append(f1_score(tps, fps, fns))


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
        confusion = init_confusion()
        with torch.no_grad():
            print(f'Start Validation Epoch {epoch+1}.')
            for batch in val_loader:
                input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # convert output to binary format
                bin_out = torch.where(torch.sigmoid(outputs) >=0.5, torch.tensor(1.0), torch.tensor(0.0))
                # collect confusion statistics:
                for i in range(num_labels):
                    for j in range(len(bin_out)):
                        if bin_out[j][i] == 1.0 and labels[j][i] == 1.0:
                            confusion[i]['tp'] += 1
                        elif bin_out[j][i] == 0.0 and labels[j][i] == 0.0:
                            confusion[i]['tn'] += 1
                        elif bin_out[j][i] == 1.0 and labels[j][i] == 0.0:
                            confusion[i]['fp'] += 1
                        elif bin_out[j][i] == 0.0 and labels[j][i] == 1.0:
                            confusion[i]['fn'] += 1
                loss = criterion(outputs, labels.float())
                total_val_loss += loss.item()
        
        # calculate confusion stats in macro and micro strategy manner
        # macro strategy
        acc = [accuracy(confusion[i]['tp'], confusion[i]['tn'], confusion[i]['fp'], confusion[i]['fn']) for i in range(num_labels)]
        prec = [precision(confusion[i]['tp'], confusion[i]['fp']) for i in range(num_labels)]
        rec = [recall(confusion[i]['tp'], confusion[i]['fn']) for i in range(num_labels)]
        f1 = [f1_score(confusion[i]['tp'], confusion[i]['fp'], confusion[i]['fn']) for i in range(num_labels)]
        # insert values in dict
        stats['val']['accuracy']['macro'].append(sum(acc)/len(acc))
        stats['val']['precision']['macro'].append(sum(prec)/len(prec))
        stats['val']['recall']['macro'].append(sum(rec)/len(rec))
        stats['val']['f1']['macro'].append(sum(f1)/len(f1))
        # micro strategy
        tps = sum([confusion[i]['tp'] for i in range(num_labels)])
        tns = sum([confusion[i]['tn'] for i in range(num_labels)])
        fps = sum([confusion[i]['fp'] for i in range(num_labels)])
        fns = sum([confusion[i]['fn'] for i in range(num_labels)])
        # insert values in dict
        stats['val']['accuracy']['micro'].append(accuracy(tps, tns, fps, fns))
        stats['val']['precision']['micro'].append(precision(tps, fps))
        stats['val']['recall']['micro'].append(recall(tps, fns))
        stats['val']['f1']['micro'].append(f1_score(tps, fps, fns))

        print("Predictions: ", torch.where(torch.sigmoid(outputs) >=0.5, torch.tensor(1.0), torch.tensor(0.0)))
        print("Labels: ", labels)
        val_losses.append(total_val_loss)
        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Validation Loss = {val_losses[-1]:.4f}")
    
   
        ### Test Model on SemEval Test Dataset ###
        model.eval()
        confusion = init_confusion()
        total_test_loss = 0
        with torch.no_grad():
            print(f'Start Testing on Test Dataset!')
            for batch in test_loader:
                input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # convert output to binary format
                bin_out = torch.where(torch.sigmoid(outputs) >=0.5, torch.tensor(1.0), torch.tensor(0.0))
                # collect confusion statistics:
                for i in range(num_labels):
                    for j in range(len(bin_out)):
                        if bin_out[j][i] == 1.0 and labels[j][i] == 1.0:
                            confusion[i]['tp'] += 1
                        elif bin_out[j][i] == 0.0 and labels[j][i] == 0.0:
                            confusion[i]['tn'] += 1
                        elif bin_out[j][i] == 1.0 and labels[j][i] == 0.0:
                            confusion[i]['fp'] += 1
                        elif bin_out[j][i] == 0.0 and labels[j][i] == 1.0:
                            confusion[i]['fn'] += 1
                loss = criterion(outputs, labels.float())
                total_test_loss += loss.item()

        test_losses.append(total_test_loss)
        # calculate confusion stats in macro and micro strategy manner
        # macro strategy
        acc = [accuracy(confusion[i]['tp'], confusion[i]['tn'], confusion[i]['fp'], confusion[i]['fn']) for i in range(num_labels)]
        prec = [precision(confusion[i]['tp'], confusion[i]['fp']) for i in range(num_labels)]
        rec = [recall(confusion[i]['tp'], confusion[i]['fn']) for i in range(num_labels)]
        f1 = [f1_score(confusion[i]['tp'], confusion[i]['fp'], confusion[i]['fn']) for i in range(num_labels)]
        # insert values in dict
        stats['test']['accuracy']['macro'].append(sum(acc)/len(acc))
        stats['test']['precision']['macro'].append(sum(prec)/len(prec))
        stats['test']['recall']['macro'].append(sum(rec)/len(rec))
        stats['test']['f1']['macro'].append(sum(f1)/len(f1))
        # micro strategy
        tps = sum([confusion[i]['tp'] for i in range(num_labels)])
        tns = sum([confusion[i]['tn'] for i in range(num_labels)])
        fps = sum([confusion[i]['fp'] for i in range(num_labels)])
        fns = sum([confusion[i]['fn'] for i in range(num_labels)])
        # insert values in dict
        stats['test']['accuracy']['micro'].append(accuracy(tps, tns, fps, fns))
        stats['test']['precision']['micro'].append(precision(tps, fps))
        stats['test']['recall']['micro'].append(recall(tps, fns))
        stats['test']['f1']['micro'].append(f1_score(tps, fps, fns))   

    # insert loss statistics
    stats['train']['loss'] = train_losses
    stats['val']['loss'] = val_losses
    stats['test']['loss'] = test_losses

    with open(os.path.join(os.path.dirname(__file__), f'../outputs/statistics/{model.name}_fold_{fold}_stats.json'), 'w') as file:
        json.dump(stats, file, indent=4)

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

        train(fold+1, cfg.epochs, cfg.num_labels, model, train_loader, val_loader, test_loader, optimizer, criterion, device)

if __name__ == '__main__':
    cross_validation()
