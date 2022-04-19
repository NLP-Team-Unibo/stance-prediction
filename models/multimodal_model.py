from cgitb import text
import torch
from torch import nn
from models.text_model import TextModel
from models.audio_model import AudioModel

class MultimodalModel(nn.Module):
    def __init__(
            self, 
            text_model, 
            audio_model, 
            p_list = (0.3),
            device='cuda'
        ):
        super(MultimodalModel, self).__init__()
        self.text_model = text_model
        self.audio_model = audio_model
        self.dropout = nn.Dropout(p=p_list[0])
        self.classifier = nn.Linear(512+768, 1)

        #self.relu = nn.ReLU()
    
    def forward(self, text_input, audio_input):
        x = self.text_model(**text_input)
        y = self.audio_model(audio_input)
        x = torch.cat([x, y], dim=1)
        x = self.dropout(x)
        x = self.classifier(x)
        #x = self.relu(x)
        return x