from transformers import RobertaModel, RobertaForSequenceClassification
import torch.nn as nn 
import torch.nn.functional as F
import torch


class RobertaForSequenceClassificationAttentionPooling(nn.Module):
    def __init__(self, backbone, num_classes):
        super(RobertaForSequenceClassificationAttentionPooling, self).__init__()
        self.name = 'RobertaForSequenceClassificationAttentionPooling'
        self.backbone = RobertaModel.from_pretrained(backbone)
        self.attention_pooling = AttentionPooling(self.backbone.config.hidden_size).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.1), 
            nn.Linear(768, num_classes)
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
        self.name = 'RobertaForSequenceClassificationMaxPooling'
        self.backbone = RobertaModel.from_pretrained(backbone)
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.1), 
            nn.Linear(768, num_classes)
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
        self.name = 'RobertaForSequenceClassificationMeanPooling'
        self.backbone = RobertaModel.from_pretrained(backbone)
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.1), 
            nn.Linear(768, num_classes)
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
        self.name = 'RobertaForSequenceClassificationDeep'
        self.backbone = RobertaModel.from_pretrained(backbone)
        self.pre_classifier = nn.Linear(768, 768)
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.1), 
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        cls = self.backbone(input_ids, attention_mask)
        cls = cls.last_hidden_state[:, 0, :]
        cls = self.classifier(cls)
        return cls


class RobertaForSequenceClassificationPure(nn.Module):
    def __init__(self, backbone, num_classes):
        super(RobertaForSequenceClassificationPure, self).__init__()
        self.name = 'RobertaForSequenceClassificationPure'
        self.backbone = RobertaModel.from_pretrained(backbone)
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.1), 
            nn.Linear(768, num_classes)
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
        context_vector = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        return context_vector

class AggregatedLayers(nn.Module):
    def __init__(self, num_layers):
        super(AggregatedLayers, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)
    
    def forward(self, all_hidden_states):
        # stack hidden states
        all_hidden_states = torch.stack(all_hidden_states, dim=0)
        # weighted sum of all layers
        weighted_layers = torch.sum( weighted_layers = torch.sum(self.weights[:, None, None, None] * all_hidden_states, dim=0), dim=0)
        return weighted_layers

class MeanMaxAttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(MeanMaxAttentionPooling, self).__init__()
        self.attention_pooling = AttentionPooling(hidden_dim).to('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, hidden_states, attention_mask):
        # max pooling first
        max_pooling = torch.max(hidden_states, dim=1)[0]
        # mean pooling
        sum_hidden = torch.sum(hidden_states, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1)
        mean_pooling = sum_hidden / sum_mask
        # attention_pooling 
        attention_pooling = self.attention_pooling(hidden_states, attention_mask)

        return torch.cat([mean_pooling, max_pooling, attention_pooling], dim=1)
    
class PoolingRoberta(nn.Module):
    def __init__(self, bacbbone, num_classes):
        super(PoolingRoberta, self).__init__()
        self.name = "PoolingRoberta"
        self.backbone = RobertaModel.from_pretrained(bacbbone)
        #self.layer_aggregation = AggregatedLayers(num_layers=4)
        self.pooling = MeanMaxAttentionPooling(hidden_dim=768).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.clasifier = nn.Sequential(
            nn.Linear(768 * 3, 768),
            nn.ReLU(),
            nn.LayerNorm(768),
            nn.Linear(768, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-4:]
        aggregated_hidden_states = torch.mean(torch.stack(hidden_states, dim=0), dim=0)
        pooled_outout = self.pooling(aggregated_hidden_states, attention_mask)
        out = self.clasifier(pooled_outout)
        print("Completed first forward pass!")
        return out