import json 
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

class EmotionData(Dataset):

    def __init__(self, file_path, model, max_length=512):

        self.data = self._load_json(file_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(model)
        self.label_mapping = {
            'anger' : 0,
            'fear' : 1,
            'joy' : 2,
            'sadness' : 3, 
            'surprise' : 4, 
            'none' : 5
        } 
        self.max_length = max_length
    

    def _load_json(self, file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    
    def _convert_labels(self, label_strings):
        binary_labels = [0] * len(self.label_mapping)
        for label in label_strings:
            if label in self.label_mapping:
                binary_labels[self.label_mapping[label]] = 1
        return binary_labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        text = sample['sentence']
        labels = self._convert_labels(sample['emotions'])

        encoding = self.tokenizer(
            text, 
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids' : encoding['input_ids'].squeeze(0),
            'attention_mask' : encoding['attention_mask'].squeeze(0),
            'labels' : torch.tensor(labels, dtype=torch.float)
        }