import torch
from torch import nn
from transformers import DistilBertModel

class TextModel(nn.Module):
    def __init__(self, classify=False):
        super(TextModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained('bert-base-uncased')
        self.classify=classify
        for param in self.bert.parameters():
            param.requires_grad = False
        for layer in self.bert.transformer.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.pre_classifier = nn.Linear(768, 768)
        self.classifier= nn.Linear(768, 1)
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = x[0][:, 0]
        x = self.pre_classifier(x)
        if self.classify:
            x = self.classifier(x)
            x = self.relu(x)
        return x