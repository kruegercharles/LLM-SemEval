from transformers import RobertaModel
import torch.nn as nn 
import torch.nn.functional as F
import torch

class RobertaMultiLabelClassification(nn.Module):
    def __init__(self, backbone, num_labels):
        super(RobertaMultiLabelClassification, self).__init__()
        self.backbone = RobertaModel.from_pretrained(backbone)
        # hidden.size = 768 for roberta-base and 1024 for roberta-large
        hidden_size = self.backbone.config.hidden_size
        self.attention_weights = nn.Parameter(torch.randn(hidden_size))
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        # forward pass of backbone
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # last hidden state of shape (batch_size, seq_len, hidden_size)
        last_hidden_state = out.last_hidden_state
        # attention score calculation -> per token
        attention_sc = torch.matmul(last_hidden_state, self.attention_weights)
        # normalize across tokens & unsqueeze -> shape (batch_size, seq_len, 1)
        attention_sc = (F.softmax(attention_sc, dim=1)).unsqueeze(-1) 
        # apply pooling
        pooled_output = torch.sum(last_hidden_state * attention_sc, dim=1)
        # pass pooled output to the classifier
        logits = self.classifier(pooled_output)
        return logits