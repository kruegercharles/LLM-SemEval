import torch 
from torch.utils.data import DataLoader, random_split
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, get_scheduler, DataCollatorWithPadding, TrainingArguments, Trainer
from models.lm_classifier import RobertaMultiLabelClassification
from datasets import load_from_disk

def tokenize(data, tokenizer):
    return tokenizer(data['text'], padding='max_length', truncation=True, max_length=512)

def prepare_data(dset_id : str, tokenizer):
    data = load_from_disk(dset_id)

    tokenized_data = data.map(lambda x: tokenize(x, tokenizer), batched=True)
    # remove text column to improve memory efficiency since tokenizer gives back input_ids and attention mask
    tokenized_data = tokenized_data.remove_columns(['text'])
    # rename 'labels' to 'label' to verify coherence to tokenizer naming scheme 
    tokenized_data = tokenized_data.rename_column('labels', 'label')
    tokenized_data.set_format('torch')

    # for training just make a train-test split
    train_size, val_size, test_size = int(0.8 * len(tokenized_data)), int(0.1 * len(tokenized_data)), int(0.1 * len(tokenized_data))
    if len(tokenized_data) != train_size+val_size+test_size:
        train_size += len(tokenized_data) - (train_size+test_size+val_size)
    print(f"len dset: {len(tokenized_data)}")
    train_data, val_data, test_data = random_split(tokenized_data, [train_size, val_size, test_size])

    return train_data, val_data, test_data

def train(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs):
    train_losses, val_losses = list(), list()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0 

        for batch in train_loader:
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            print(f'Outputs: {outputs}, Labels: {labels}')
            loss = criterion(outputs, labels.float())
            print(f'Loss: {loss}')
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        train_losses.append(total_train_loss / len(train_loader))

        # Validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels.float())
                total_val_loss += loss.item()

        val_losses.append(total_val_loss / len(val_loader))
        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")

        return train_losses, val_losses
    
def initialize_weights_xavier(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()
    
if __name__ == '__main__':
    device = 'cpu'
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    train_dataset, val_dataset, test_dataset = prepare_data('/Users/eliaruhle/Documents/LLM-SemEval/data/codabench_data/train/combined', tokenizer)

    model = RobertaMultiLabelClassification('roberta-base', num_labels=7).to(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=1000, num_training_steps=len(train_dataset))
    criterion = nn.BCEWithLogitsLoss().to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors=device)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    num_epochs = 2
    print("Starting Training")
    train_losses, val_losses = train(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs)

    plot_losses(train_losses, val_losses)