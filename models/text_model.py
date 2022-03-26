import torch
from torch import nn
from transformers import BertModel

class TextModel(nn.Module):
    def __init__(self, ):
        super(TextModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        """ for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.bert.pooler.dense.parameters():
            param.requires_grad = True """
        self.linear1 = nn.Linear(768, 768)
        self.linear2 = nn.Linear(768, 1)
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, segment_tensors, attention_mask):
        x = self.bert(input_ids=input_ids, token_type_ids=segment_tensors, attention_mask=attention_mask).pooler_output
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x