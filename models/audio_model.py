import torch
from torch import nn
import torchaudio
from models.stance_prediction_module import StancePredictionModule

class AudioModel(StancePredictionModule):
    def __init__(
        self, 
        chunk_length=10, 
        n_transformers=12,
        n_trainable_layers=2,
        p_list=(0.3, 0.3),
        pre_classifier=True,
        classify=False
    ):

        super(AudioModel, self).__init__()
        assert n_transformers >= n_trainable_layers, 'Number of transformer layers must be greater or equal to the number of trainable layers'
        self.chunk_size = chunk_length
        self.classify = classify
        self.__bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec2 = self.__bundle.get_model()
        self.wav2vec2.encoder.transformer.layers = self.wav2vec2.encoder.transformer.layers[:n_transformers]
        
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        if n_trainable_layers > 0:
            for layer in self.wav2vec2.encoder.transformer.layers[-n_trainable_layers:]:
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
        
    
    def forward(self, audio):
        x, _ = self.wav2vec2(audio)
        x = torch.mean(x, dim=1)
        x = self.dropout1(x)
        if self.pre_classifier:
            x = self.pre_classifier(x)
            x = self.dropout2(x)
        x = self.relu(x)
        if self.classify:
            x = self.classifier(x)
        return x