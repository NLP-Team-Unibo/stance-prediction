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
        classify=False,
        return_sequences=False
    ):
        """
            Creates the desired model for the text classification task. It is based on the huggingface implementation of DistilBert, which can then
            be followed by one or two linear layers. 
            Note that DistilBert adds a [CLS] token at the begginning of each input and that its output is inferred by all the other words in the 
            sentence. Thus, for our text classification task we are only interested in using this embedding and it is the only input we give to the 
            linear layers.
    
            Parameters
            ----------
            distilbert_type: str
                Either 'distilbert-base-uncased' or 'distilbert-base-cased', depending on whether we want to use the cased or uncased version
                of DistilBert and its tokenizer. Default to 'distilbert-base-uncased'.
            n_trainable_layers: int
                Number of transformer layers to fine-tune. All the layers before are fixed to pre-trained weights. Default to 2.
            dropout_values: tuple of floats
                The values to be used for dropout, respectively before and after the pre-classifier part of the network. Default to (0.3, 0.3).
            pre_classifier: bool
                Whether to add or not another linear layer before the final one. Default to True.
            classify: bool
                Whether to add or not the final layer, which takes care of making the prediction. This will typically be set to
                False only when the TextModel is used as a component in the bigger MultimodalModel. Default to True.
            
        """
        super(TextModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained(distilbert_type)
        self.bert_out_dim = self.bert.transformer.layer[-1].output_layer_norm.normalized_shape[0]
        self.classify = classify
        self.pre_classifier = pre_classifier
        self.return_sequences = return_sequences

        # Make all the DistilBert parameters non-trainable
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Make it possible to fine-tune the last n_trainable_layers of the transformer.
        if n_trainable_layers > 0:
            for layer in self.bert.transformer.layer[-n_trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        

        if not self.return_sequences:
            self.dropout1 = nn.Dropout(p=dropout_values[0])
            self.relu = nn.ReLU()
            if pre_classifier:
                self.pre_classifier = nn.Linear(self.bert_out_dim, self.bert_out_dim)
                self.dropout2 = nn.Dropout(p=dropout_values[1])
        
            if classify:
                self.classifier= nn.Linear(self.bert_out_dim, 1)
            
    
    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # DistilBert returns a sequence but we are only interested in working with the first element.
        if self.return_sequences:
            return x[0]
        x = x[0][:, 0]

        # Classification
        x = self.dropout1(x)
        
        if self.pre_classifier:
            x = self.pre_classifier(x)
            x = self.dropout2(x)
        x = self.relu(x)

        if self.classify:
            x = self.classifier(x)
        return x