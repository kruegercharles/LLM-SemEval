from transformers import RobertaModel
import torch.nn as nn 

class RobertaMultiLabelClassification(nn.Module):
    def __init__(self, backbone, num_labels):
        super(RobertaMultiLabelClassification, self).__init__()
        self.backbone = RobertaModel.from_pretrained(backbone)
        # hidde._size = 768 for roberta-base and 1024 for roberta-large
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
        # Forward pass of roberta Backbone
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Take the first CLS token from the last hidden state
        pooled_output = out.last_hidden_state[:, 0, :] # shape (batch_size, hidden_size)

        # pass pooled output to the classifier
        logits = self.classifier(pooled_output)
        return logits