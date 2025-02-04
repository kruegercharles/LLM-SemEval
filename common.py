import re

import torch
import torch.nn as nn
from transformers import RobertaModel


def remove_junk(line: str) -> str:
    line = line.replace("\n", "")
    line = line.replace("~", "")
    line = line.replace("`", "")
    line = line.replace("<", "")
    line = line.replace(">", "")
    line = line.replace(";", "")
    line = line.replace("&", "")
    line = line.replace("#", "")
    line = line.replace("nbsp;", "")
    line = line.replace("*", "")
    line = line.replace("''", "")
    line = line.replace("{", "")
    line = line.replace("}", "")
    line = line.replace("[NAME]", "")

    while line.count("  ") > 0:
        line = line.replace("  ", " ")

    # remove urls
    while re.search(r"http\S+", line):
        line = re.sub(r"http\S+", "", line)

    while re.search(r"www\.\[a-zA-Z0-9]+", line):
        line = re.sub(r"www\.\[a-zA-Z0-9]+", "", line)

    # remove all words that start with @
    while re.search(r"@\S+", line):
        line = re.sub(r"@\S+", "", line)

    # remove still like \ud83d or \ude0d or similar
    # while re.search(r"\\u\S+", line):
    # line = re.sub(r"\\u\S+", "", line)

    # remove all words that start with \
    while re.search(r"\\\S+", line):
        line = re.sub(r"\\\S+", "", line)

    # remove \r
    while re.search(r"\\r", line):
        line = re.sub(r"\\r", "", line)

    # if there is only one " in the line, remove it
    if line.count('"') == 1:
        line = line.replace('"', "")

    if line.endswith(":"):
        line = line[:-1]

    # remove spaces at the beginning and end
    line = line.strip()

    return line


EMOTION_LABELS = [
    "anger",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "disgust",
    "none",
]


# => there is no disgust in the dataset
EMOTION_COMPLEX_LABELS = [
    "light anger",
    "medium anger",
    "strong anger",
    "light fear",
    "medium fear",
    "strong fear",
    "light joy",
    "medium joy",
    "strong joy",
    "light sadness",
    "medium sadness",
    "strong sadness",
    "light surprise",
    "medium surprise",
    "strong surprise",
    "none",
]


def get_precision(tp: int, fp: int) -> float:
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def get_recall(tp: int, fn: int) -> float:
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def get_f1_score(tp: int, fp: int, fn: int) -> float:
    precision = get_precision(tp, fp)
    recall = get_recall(tp, fn)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def get_accuracy(tp: int, fp: int, tn: int, fn: int) -> float:
    if tp + fp + tn + fn == 0:
        return 0
    return (tp + tn) / (tp + fp + tn + fn)


class RobertaForSequenceClassificationAttentionPooling(nn.Module):
    def __init__(self, backbone, num_classes):
        super(RobertaForSequenceClassificationAttentionPooling, self).__init__()
        self.name = "RobertaForSequenceClassificationAttentionPooling"
        self.backbone = RobertaModel.from_pretrained(backbone)
        self.attention_pooling = AttentionPooling(self.backbone.config.hidden_size).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 768), nn.Dropout(0.1), nn.Linear(768, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        cls = self.backbone(input_ids, attention_mask)
        cls = cls.last_hidden_state
        cls = self.attention_pooling(cls, attention_mask)
        cls = self.classifier(cls)
        return cls


class RobertaForSequenceClassificationMaxPooling(nn.Module):
    def __init__(self, backbone, num_classes):
        super(RobertaForSequenceClassificationMaxPooling, self).__init__()
        self.name = "RobertaForSequenceClassificationMaxPooling"
        self.backbone = RobertaModel.from_pretrained(backbone)
        self.classifier = nn.Sequential(
            nn.Linear(768, 768), nn.Dropout(0.1), nn.Linear(768, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        cls = self.backbone(input_ids, attention_mask)
        cls = cls.last_hidden_state
        cls = self.max_pooling(cls, attention_mask)
        cls = self.classifier(cls)
        return cls

    def max_pooling(self, hidden_states, attention_mask):
        mask_exp = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        hidden_states[mask_exp == 0] = -1e9
        return torch.max(hidden_states, dim=1)[0]


class RobertaForSequenceClassificationMeanPooling(nn.Module):
    def __init__(self, backbone, num_classes):
        super(RobertaForSequenceClassificationMeanPooling, self).__init__()
        self.name = "RobertaForSequenceClassificationMeanPooling"
        self.backbone = RobertaModel.from_pretrained(backbone)
        self.classifier = nn.Sequential(
            nn.Linear(768, 768), nn.Dropout(0.1), nn.Linear(768, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        cls = self.backbone(input_ids, attention_mask)
        cls = cls.last_hidden_state
        cls = self.mean_pooling(cls, attention_mask)
        cls = self.classifier(cls)
        return cls

    def mean_pooling(self, hidden_states, attention_mask):
        mask_exp = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        sum_embeddings = torch.sum(hidden_states * mask_exp, dim=1)
        sum_mask = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


class RobertaForSequenceClassificationDeep(nn.Module):
    def __init__(self, backbone, num_classes):
        super(RobertaForSequenceClassificationDeep, self).__init__()
        self.name = "RobertaForSequenceClassificationDeep"
        self.backbone = RobertaModel.from_pretrained(backbone)
        self.pre_classifier = nn.Linear(768, 768)
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.1),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        cls = self.backbone(input_ids, attention_mask)
        cls = cls.last_hidden_state[:, 0, :]
        cls = self.classifier(cls)
        return cls


class RobertaForSequenceClassificationPure(nn.Module):
    def __init__(self, backbone, num_classes):
        super(RobertaForSequenceClassificationPure, self).__init__()
        self.name = "RobertaForSequenceClassificationPure"
        self.backbone = RobertaModel.from_pretrained(backbone)
        self.classifier = nn.Sequential(
            nn.Linear(768, 768), nn.Dropout(0.1), nn.Linear(768, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        cls = self.backbone(input_ids, attention_mask)
        cls = cls.last_hidden_state[:, 0, :]
        cls = self.classifier(cls)
        return cls


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        scores = self.attention_weights(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=1)
        context_vector = torch.sum(
            hidden_states * attention_weights.unsqueeze(-1), dim=1
        )
        return context_vector
