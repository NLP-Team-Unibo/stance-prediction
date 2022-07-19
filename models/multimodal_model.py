import copy
import torch
from torch import nn

from models.stance_prediction_module import StancePredictionModule
from models.mult_modules.transformer import TransformerEncoder

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

class MultimodalModel(StancePredictionModule):
    def __init__(
            self, 
            text_model, 
            audio_model, 
            dropout_values = (0.3),
            freeze_text = False,
            freeze_audio = False,
        ):
        """
            Creates a model accepting two inputs of different type: a text sequence, which is passed to the text_model component, and
            a raw audio signal, which is fed as input to the audio_model. Once we obtain two separate embeddings for the inputs, they are 
            concatenated and passed through a linear layer for the classification step.
    
            Parameters
            ----------
            text_model: nn.Module
                The desired TextModel instance; it will be used to process the text portion of the input.
            audio_model: nn.Module
                The desired AudioModel instance; it will be used to process the audio portion of the input.
            dropout_values: tuple of floats
                The value to be used for dropout after concatenating the embeddings of the two modlities. Default to (0.3). 
            freeze_text: bool
                Whether to freeze all the parameters in the text model. Default to False.
            freeze_audio: bool
                Whether to freeze all the parameters in the audio model. Default to False.
            
        """
        super(MultimodalModel, self).__init__()
        self.text_model = text_model
        self.audio_model = audio_model

        if freeze_text: freeze_model(self.text_model)
        if freeze_audio: freeze_model(self.audio_model)
        
        self.dropout = nn.Dropout(p=dropout_values[0])
        self.classifier = nn.Linear(self.text_model.bert_out_dim + self.audio_model.wav2vec2_out_dim, 1)
    
    def forward(self, text_input, audio_input):
        x = self.text_model(**text_input)
        y = self.audio_model(audio_input)
        x = torch.cat([x, y], dim=1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

class MulT(StancePredictionModule):
    def __init__(
            self, 
            text_model, 
            audio_model, 
            dropout_values = (0.3),
            freeze_text = False,
            freeze_audio = False,
            crossmodal_type = 'audio2text', # ['audio2text', 'text2audio', 'both']
            pool_operation = 'avg', # ['avg', 'first', 'last']
        ):
        """
            Creates a model accepting two inputs of different type: a text sequence, which is passed to the text_model component, and
            a raw audio signal, which is fed as input to the audio_model. Once we obtain two separate embeddings for the inputs, they are 
            concatenated and passed through a linear layer for the classification step.
    
            Parameters
            ----------
            text_model: nn.Module
                The desired TextModel instance; it will be used to process the text portion of the input.
            audio_model: nn.Module
                The desired AudioModel instance; it will be used to process the audio portion of the input.
            dropout_values: tuple of floats
                The value to be used for dropout after concatenating the embeddings of the two modlities. Default to (0.3). 
            freeze_text: bool
                Whether to freeze all the parameters in the text model. Default to False.
            freeze_audio: bool
                Whether to freeze all the parameters in the audio model. Default to False.
            
        """
        super(MulT, self).__init__()
        self.text_model = text_model
        self.audio_model = audio_model

        if freeze_text: freeze_model(self.text_model)
        if freeze_audio: freeze_model(self.audio_model)
        
        self.crossmodal_type = crossmodal_type
        self.crossmodal = nn.ModuleList([TransformerEncoder(embed_dim=768, 
                                                num_heads=8, 
                                                layers=4, 
                                                attn_dropout=0.1, 
                                                relu_dropout=0.1, 
                                                res_dropout=0.1,
                                                embed_dropout=0.25,
                                                attn_mask=True)])

        if crossmodal_type == 'both':
            self.crossmodal.append(copy.deepcopy(self.crossmodal[0]))
            
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(1536 if crossmodal_type == 'both' else 768, 1)
        self.dropout = nn.Dropout(p=dropout_values[0])
        if pool_operation == 'avg':
            self.pool_operation = lambda x, dim: torch.mean(x, dim=dim)
        elif pool_operation == 'last':
            self.pool_operation = lambda x, _ : x[:,-1,:]
        elif pool_operation == 'first':
            self.pool_operation = lambda x, _ : x[:, 0,:]
        
    def forward(self, text_input, audio_input):
        text_sequences = self.text_model(**text_input)
        audio_sequences = self.audio_model(audio_input)
        text_sequences = text_sequences.permute(1, 0, 2)
        audio_sequences = audio_sequences.permute(1, 0, 2)
        
        #TODO check dropout
        audio_sequences = self.dropout(audio_sequences)
        text_sequences = self.dropout(text_sequences)

        if self.crossmodal_type == 'audio2text' or 'both':
            x = self.crossmodal[0](audio_sequences, text_sequences, text_sequences)
            x = x.permute(1, 0, 2)
            x = self.pool_operation(x, 1)
            #x = torch.mean(x, dim=1)
        elif self.crossmodal_type == 'text2audio':
            x = self.crossmodal[0](text_sequences, audio_sequences, audio_sequences)
            x = x.permute(1, 0, 2)
            x = self.pool_operation(x, 1)

        if len(self.crossmodal) > 1:
            y = self.crossmodal[1](text_sequences, audio_sequences, audio_sequences)
            y = y.permute(1, 0, 2)
            y = self.pool_operation(y, 1)
            x = torch.cat([x, y], dim=1)
        x = self.relu(x)       
        x = self.classifier(x)
        return x


"""from models.text_model_mult import TextModel
from models.audio_model_mult import AudioModel
t = TextModel(return_sequences=True)
a = AudioModel()

m = MultimodalModelMulT(t, a)

import torch
input_id = torch.randint(0, 10, (8, 512))
attention_mask = torch.ones((8, 512))

wav = torch.rand((8, 15*16000))

text = {'input_ids':input_id, 'attention_mask': attention_mask}
x = m(text, wav)
print(x.size())"""