import torch
from torch import nn

from models.stance_prediction_module import StancePredictionModule

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