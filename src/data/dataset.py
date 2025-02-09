import json 
import torch
import os
import re
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

class EmotionData(Dataset):

    def __init__(self, file_path, model, mapping, max_length=512):

        self.data = self._load_json(file_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(model)
        if mapping == 'full':  
            self.label_mapping = {
                'anger' : 0,
                'fear' : 1,
                'joy' : 2,
                'sadness' : 3, 
                'surprise' : 4,
                'disgust' : 5, 
                'none' : 6
            }
        else:
            self.label_mapping = {
                'anger' : 0,
                'fear' : 1,
                'joy' : 2,
                'sadness' : 3, 
                'surprise' : 4,
                'none' : 5, 
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


class IntensityData(Dataset):

    def __init__(self, file_path, model, max_length=512):
        self.data = self._load_json(file_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(model)
        self.label_mapping = {
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

class EmotionDataAdvanced(Dataset):

    def __init__(self, file_path, model, mapping, max_length=512):

        self.data = self._load_json(file_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(model)
        if mapping == 'full':  
            self.label_mapping = {
                'anger' : 0,
                'fear' : 1,
                'joy' : 2,
                'sadness' : 3, 
                'surprise' : 4,
                'disgust' : 5, 
                'none' : 6
            }
        else:
            self.label_mapping = {
                'anger' : 0,
                'fear' : 1,
                'joy' : 2,
                'sadness' : 3, 
                'surprise' : 4,
                'none' : 5, 
            }
        self.max_length = max_length
        self.lex = self._build_lex()
    

    def _load_json(self, file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    
    def _convert_labels(self, label_strings):
        binary_labels = [0] * len(self.label_mapping)
        for label in label_strings:
            if label in self.label_mapping:
                binary_labels[self.label_mapping[label]] = 1
        return binary_labels
    
    def _build_lex(self):
        path_to_lex = os.path.join(os.path.dirname(__file__), 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
        lex = {}
        with open(path_to_lex, "r") as file:
            data = file.readlines()
            current_word = ""
            idx = []
            for line in data:
                # word, emotion, present
                word, emotion, present = line.split("\t")
                if current_word != word:
                    # save entry
                    if idx != []:
                        lex[current_word] = idx
                    idx = []
                    current_word = word
                if emotion in self.label_mapping.keys() and int(present[0]) == 1:
                    idx.append(self.label_mapping[emotion])
        return lex

    def _text_to_lex_vec(self, text:str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        vec = [0.0]*(len(self.label_mapping.keys())-1)
        for w in text:
            if w in self.lex.keys():
                for i in self.lex[w]:
                    vec[i] += 1.0
        return vec

    
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

        vec = self._text_to_lex_vec(text)
        
        return {
            'input_ids' : encoding['input_ids'].squeeze(0),
            'attention_mask' : encoding['attention_mask'].squeeze(0),
            'labels' : torch.tensor(labels, dtype=torch.float),
            'vec' : torch.tensor(vec, dtype=torch.float)
        }

def build_lex():

    path_to_lex = os.path.join(os.path.dirname(__file__), 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    lex = {}
    label_mapping = {
                'anger' : 0,
                'fear' : 1,
                'joy' : 2,
                'sadness' : 3, 
                'surprise' : 4,
                'none' : 5, 
            }
    with open(path_to_lex, "r") as file:
        data = file.readlines()
        current_word = ""
        idx = []
        for line in data:
            # word, emotion, present
            word, emotion, present = line.split("\t")
            if current_word != word:
                # save entry
                if idx != []:
                    lex[current_word] = idx
                idx = []
                current_word = word
            if emotion in ["anger", "fear",  "joy", "sadness", "surprise"] and int(present[0]) == 1:
                print(emotion)
                idx.append(label_mapping[emotion])
                print(idx)
    
    print(lex)

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    
    # Removing special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Splitting into words
    words = text.split()
    
    return words
