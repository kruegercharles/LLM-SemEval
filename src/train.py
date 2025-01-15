import torch 
import torch.nn as nn 
import torch.optim as optim 
import hydra
import os
from torch.utils.data import DataLoader, random_split
from transformers import RobertaTokenizer, get_scheduler, DataCollatorWithPadding, TrainingArguments, Trainer
from models.lm_classifier import RobertaMultiLabelClassification
from datasets import load_from_disk
from data.preprocessing import one_hot_encoding, retrieve_datasets
from omegaconf import DictConfig
from misc.misc import search_available_devices, plot_losses, statistics_to_csv, count_correct_samples
from sklearn.metrics import f1_score

def tokenize(data, tokenizer):
    # It's possible to adjust max_length or use dynamic patterns -> might be memory efficient since we rarely to never have 512 (or more) tokens per sentence
    return tokenizer(data['text'], padding='max_length', truncation=True, max_length=512)

def prepare_data(dset_id : str, tokenizer, num_labels):
    if os.path.exists(dset_id):
        # load the data from the local disk (precompiled)
        data = load_from_disk(dset_id)
    else:
        data = one_hot_encoding(retrieve_datasets(), num_labels)
        data.save_to_disk(os.path.join(os.path.dirname(__file__), f'../data/codabench_data/train/combined'))

    tokenized_data = data.map(lambda x: tokenize(x, tokenizer), batched=True)
    # rename 'labels' to 'label' to verify coherence to tokenizer naming scheme 
    tokenized_data = tokenized_data.rename_column('labels', 'label')
    tokenized_data.set_format('torch')

    # for training just make a train-test split
    train_size, val_size, test_size = int(0.8 * len(tokenized_data)), int(0.2 * len(tokenized_data)), int(0.0 * len(tokenized_data))
    if len(tokenized_data) != train_size+val_size+test_size:
        train_size += len(tokenized_data) - (train_size+test_size+val_size)
    print(f"len dset: {len(tokenized_data)}")
    train_data, val_data, test_data = random_split(tokenized_data, [train_size, val_size, test_size])

    return train_data, val_data, test_data

@hydra.main(config_path='../configs', config_name='train_roberta')
def train(cfg: DictConfig):

    # search for available devices
    device = search_available_devices()
    print(f'*** Found the following device: {device}. Start configuting training.')
    # define used tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(cfg.model)
    # retrieve datasets for training and validation
    train_dataset, val_dataset, test_dataset = prepare_data(os.path.join(os.path.dirname(__file__), cfg.data), tokenizer, cfg.num_labels)
    # define model 
    model = RobertaMultiLabelClassification(cfg.model, cfg.num_labels).to(device=device)
    # define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    # get a scheduler for learning rate adjustement
    scheduler = get_scheduler(cfg.scheduler, optimizer=optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=len(train_dataset))
    # define a loss function (BCEWithLogitsLoss for multi-label classification)
    criterion = nn.BCEWithLogitsLoss().to(device=device)
    # get data-collator to prepare torch dataloaders -> optional
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors=device)

    # get train and validation Data Loader
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)

    # define accumulators for loss after each period
    train_losses, val_losses, train_acc, val_acc, f1_train, f1_val = list(), list(), list(), list(), list(), list()

    for epoch in range(cfg.epochs):
        print(f'Start Training Epoch {epoch+1}.')
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        f1_t = []
        for batch in train_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            scheduler.step()
            f1_t.append(f1_score(labels, outputs.numpy(), average='macro', zero_division='warn'))
            correct_train += count_correct_samples(outputs, labels)
            total_train_loss += loss.item()

        train_losses.append(total_train_loss / len(train_loader))
        train_acc.append(correct_train / len(train_loader))
        f1_train.append(sum(f1_t)/len(f1_t))

        # save model checkpoint:
        checkpoint = {
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss' : total_train_loss
        }

        torch.save(checkpoint, os.path.join(os.path.dirname(__file__), f'../outputs/models/roberta_epoch_{epoch}.pth'))

        # Validation step
        model.eval()
        total_val_loss = 0
        correct_val = 0
        f1_v = []
        with torch.no_grad():
            print(f'Start Validation Epoch {epoch+1}.')
            for batch in val_loader:
                input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                if epoch == cfg.epochs:
                    print(f'Model Prediction: {torch.where(nn.Sigmoid(outputs) >=0.5, torch.tensor(1), torch.tensor(0))}')
                    print(f'Ground Truth: {labels}')
                    text = batch['text']
                    print(f'For the given texts: {text}')
                loss = criterion(outputs, labels.float())
                correct_val += count_correct_samples(outputs, labels)
                f1_v.append(f1_score(labels, outputs.numpy(), average='macro', zero_division='warn'))
                total_val_loss += loss.item()

        val_losses.append(total_val_loss / len(val_loader))
        val_acc.append(correct_val / len(val_loader))
        f1_val.append(sum(f1_v)/len(f1_v))
        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Validation Loss = {val_losses[-1]:.4f}")

    statistics_to_csv(os.path.join(os.path.dirname(__file__), '../outputs/statistics/train_loss_final.csv'), train_losses)
    statistics_to_csv(os.path.join(os.path.dirname(__file__), '../outputs/statistics/validation_loss_final.csv'), val_losses)
    statistics_to_csv(os.path.join(os.path.dirname(__file__), '../outputs/statistics/train_acc_final.csv'), train_acc)
    statistics_to_csv(os.path.join(os.path.dirname(__file__), '../outputs/statistics/validation_acc_final.csv'), val_acc)
    statistics_to_csv(os.path.join(os.path.dirname(__file__), '../outputs/statistics/train_f1_final.csv'), f1_train)
    statistics_to_csv(os.path.join(os.path.dirname(__file__), '../outputs/statistics/validation_f1_final.csv'), f1_val)
    
if __name__ == '__main__':
    train()