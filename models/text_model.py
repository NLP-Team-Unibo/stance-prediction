import torch
from torch import nn
from transformers import DistilBertModel

class TextModel(nn.Module):
    def __init__(
        self, 
        distilbert_type='distilbert-base-uncased',
        n_trainable_layers=2,
        p_list=(0.3, 0.3),
        pre_classifier=True,
        classify=False
    ):
        super(TextModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained(distilbert_type)
        self.classify=classify
        for param in self.bert.parameters():
            param.requires_grad = False
        for layer in self.bert.transformer.layer[-n_trainable_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.dropout1 = nn.Dropout(p=p_list[0])
        self.pre_classifier = pre_classifier
        if pre_classifier:
            self.pre_classifier = nn.Linear(768, 768)
            self.dropout2 = nn.Dropout(p=p_list[1])
        self.relu = nn.ReLU()
        if classify:
            self.classifier= nn.Linear(768, 1)
            
    
    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = x[0][:, 0]
        x = self.dropout1(x)
        if self.pre_classifier:
            x = self.pre_classifier(x)
            x = self.dropout2(x)
        x = self.relu(x)
        if self.classify:
            x = self.classifier(x)
        return x