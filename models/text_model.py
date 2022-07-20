from turtle import forward
from torch import nn
from transformers import DistilBertModel, BartTokenizer
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



import torch
from torchinfo import summary


from transformers import BartForConditionalGeneration

from transformers import BartForConditionalGeneration
from models.multimodal_bart.model import MultiModalBartForConditionalGeneration
from models.multimodal_bart.config import MultiModalBartConfig

class BartTextModel(StancePredictionModule):
    def __init__(
        self, 
        bart_type='facebook/bart-base',
        n_trainable_layers=2,
        dropout_values=(0.3, 0.3),
        pre_classifier=True,
        classify=False,
        return_sequences=False,
        multimodal=False
    ):
        super(BartTextModel, self).__init__()
        self.multimodal = multimodal
        if multimodal:
            cfg = MultiModalBartConfig()
            self.text_gen = MultiModalBartForConditionalGeneration.from_pretrained(bart_type, config=cfg)
        else:
            self.text_gen = BartForConditionalGeneration.from_pretrained(bart_type)
        self.shared_base_model = getattr(self.text_gen, 'model')
        self.bert_out_dim = self.shared_base_model.decoder.layers[-1].final_layer_norm.normalized_shape[0]
        self.classify = classify
        self.pre_classifier = pre_classifier
        self.return_sequences = return_sequences

        # Make all the DistilBert parameters non-trainable
        for param in self.shared_base_model.parameters():
            param.requires_grad = False

        for param in self.shared_base_model.shared.parameters():
            param.requires_grad = True
        
        for param in self.shared_base_model.get_encoder().embed_audio.parameters():
            param.requires_grad = True
        
        # Make it possible to fine-tune the last n_trainable_layers of the transformer.
        #if n_trainable_layers > 0:
        #    for layer in self.shared_base_model.decoder.layers[-n_trainable_layers:]:
        #        for param in layer.parameters():
        #            param.requires_grad = True
        

        if not self.return_sequences:
            self.dropout1 = nn.Dropout(p=dropout_values[0])
            self.relu = nn.ReLU()
            if pre_classifier:
                self.pre_classifier = nn.Linear(self.bert_out_dim, self.bert_out_dim)
                self.dropout2 = nn.Dropout(p=dropout_values[1])
        
            if classify:
                self.classifier= nn.Linear(self.bert_out_dim, 1)
            
    
    def forward(self, input_ids, audio_features=None,  attention_mask=None, labels=None, decoder_input_ids=None, decoder_attention_mask=None):
        if self.multimodal:
            x = self.shared_base_model(input_ids=input_ids, audio_features=audio_features, attention_mask=attention_mask)
        else:
            x = self.shared_base_model(input_ids=input_ids, attention_mask=attention_mask)

        # DistilBert returns a sequence but we are only interested in working with the first element.
        if self.return_sequences:
            return x[0]

        x = x[0][:,-1, :]  # last hidden state

        # Classification
        x = self.dropout1(x)
        
        if self.pre_classifier:
            x = self.pre_classifier(x)
            x = self.dropout2(x)
        x = self.relu(x)

        if self.classify:
            x = self.classifier(x)

        if self.multimodal:
            text_gen = self.text_gen(input_ids=input_ids, 
            audio_features=audio_features, attention_mask=attention_mask, labels=labels)
        else:
            text_gen = self.text_gen(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return x, text_gen
    
    def generate(self, **kwargs):
        return self.text_gen.generate(**kwargs)

"""

from collections import OrderedDict
def transform_state_dict(distilbert_state_dict):
    pairs = []
    for (key, val) in distilbert_state_dict.items():
        new_key = key.replace('transformer', 'encoder')
        new_key = new_key.replace('q_lin', 'self.query')
        new_key = new_key.replace('k_lin', 'self.key')
        new_key = new_key.replace('v_lin', 'self.value')
        new_key = new_key.replace('out_lin', 'output.dense')
        new_key = new_key.replace('sa_layer_norm', 'attention.output.LayerNorm')
        new_key = new_key.replace('ffn.lin1', 'intermediate.dense')
        new_key = new_key.replace('ffn.lin2', 'output.dense')
        new_key = new_key.replace('output_layer_norm', 'output.LayerNorm')
        pairs.append([new_key, val])
    new_state_dict = OrderedDict(pairs)
    return new_state_dict

from transformers.models.bert.modeling_bert import BertModel
from torchinfo import summary
class TextGeneration(StancePredictionModule):
    def __init__(self):
        super(TextGeneration, self).__init__()
        self.bert_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        self.bert_decoder = BertModel.from_pretrained('bert-base-uncased').encoder
        self.bert_decoder.config.is_decoder = True
        self.bert_decoder.config.add_cross_attention = True

        
        self.bert_decoder.layer = self.bert_decoder.layer[:6]
        
        new_state_dict = transform_state_dict(self.bert_decoder.state_dict())
        self.bert_decoder.load_state_dict(new_state_dict)
    
    def forward(self, input_ids, attention_mask):
        x = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
        print(x[0].size())
        x = x[0]
        return self.bert_decoder(x)

model = TextGeneration()
summary(model)
ids = torch.ones(size=(8, 512), dtype=torch.int32)
a_mask = torch.ones(size=(8, 512))

y = model(ids, a_mask)
print(y[0].shape)
"""