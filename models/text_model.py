from torch import nn
from transformers import DistilBertModel
from models.stance_prediction_module import StancePredictionModule

class TextModel(StancePredictionModule):
    def __init__(
        self, 
        distilbert_type='distilbert-base-uncased',
        n_trainable_layers=2,
        dropout_values=(0.3, 0.3),
        pre_classifier=True,
        classify=False
    ):
        super(TextModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained(distilbert_type)
        self.bert_out_dim = self.bert.transformer.layer[-1].output_layer_norm.normalized_shape[0]
        self.classify=classify
        for param in self.bert.parameters():
            param.requires_grad = False
        if n_trainable_layers > 0:
            for layer in self.bert.transformer.layer[-n_trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        self.dropout1 = nn.Dropout(p=dropout_values[0])
        self.pre_classifier = pre_classifier
        if pre_classifier:
            self.pre_classifier = nn.Linear(self.bert_out_dim, self.bert_out_dim)
            self.dropout2 = nn.Dropout(p=dropout_values[1])
        self.relu = nn.ReLU()
        if classify:
            self.classifier= nn.Linear(self.bert_out_dim, 1)
            
    
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