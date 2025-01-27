from transformers import RobertaModel
import torch.nn as nn 
import torch.nn.functional as F
import torch


class RobertaForSequenceClassificationAttentionPooling(nn.Module):
    def __init__(self, backbone, num_classes):
        super(RobertaForSequenceClassificationAttentionPooling, self).__init__()
        self.backbone = RobertaModel.from_pretrained(backbone)
        self.attention_pooling = AttentionPooling(self.backbone.hidden_size).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.pre_classifier = nn.Linear(768, 768)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        cls = self.backbone(input_ids, attention_mask)
        cls = cls.last_hidden_state
        cls = self.attention_pooling(cls, attention_mask)
        cls = self.pre_classifier(cls)
        cls = self.classifier 
        return 

class RobertaForSequenceClassificationMaxPooling(nn.Module):
    def __init__(self, backbone, num_classes):
        super(RobertaForSequenceClassificationMaxPooling, self).__init__()
        self.backbone = RobertaModel.from_pretrained(backbone)
        self.pre_classifier = nn.Linear(768, 768)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        cls = self.backbone(input_ids, attention_mask)
        cls = cls.last_hidden_state
        cls = self.mean_pooling(cls, attention_mask)
        cls = self.pre_classifier(cls)
        cls = self.classifier 
        return 
    
    def max_pooling(self, hidden_states, attention_mask):
        mask_exp = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        hidden_states[mask_exp == 0] = -1e9
        return torch.max(hidden_states, dim=1)[0]

class RobertaForSequenceClassificationMeanPooling(nn.Module):
    def __init__(self, backbone, num_classes):
        super(RobertaForSequenceClassificationMeanPooling, self).__init__()
        self.backbone = RobertaModel.from_pretrained(backbone)
        self.pre_classifier = nn.Linear(768, 768)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        cls = self.backbone(input_ids, attention_mask)
        cls = cls.last_hidden_state
        cls = self.mean_pooling(cls, attention_mask)
        cls = self.pre_classifier(cls)
        cls = self.classifier 
        return 
    
    def mean_pooling(self, hidden_states, attention_mask):
        mask_exp = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        sum_embeddings = torch.sum(hidden_states * mask_exp, dim=1)
        sum_mask = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

class RobertaForSequenceClassificationDeep(nn.Module):
    def __init__(self, backbone, num_classes):
        super(RobertaForSequenceClassificationDeep, self).__init__()
        self.backbone = RobertaModel.from_pretrained(backbone)
        self.pre_classifier = nn.Linear(768, 768)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        cls = self.backbone(input_ids, attention_mask)
        cls = cls.last_hidden_state[:, 0, :]
        cls = self.pre_classifier(cls)
        cls = self.classifier(cls)
        return cls


class RobertaForSequenceClassification(nn.Module):
    def __init__(self, backbone, num_classes):
        super(RobertaForSequenceClassification, self).__init__()
        self.backbone = RobertaModel.from_pretrained(backbone)
        self.pre_classifier = nn.Linear(768, 768)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(768, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        cls = self.backbone(input_ids, attention_mask)
        cls = cls.last_hidden_state[:, 0, :]
        cls = self.pre_classifier(cls)
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
        context_vector = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        return context_vector
