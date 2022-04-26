import torch
from torch import nn
import torchaudio
from models.stance_prediction_module import StancePredictionModule

class AudioModel(StancePredictionModule):
    def __init__(
        self, 
        chunk_length=10, 
        downsampler_out_dim=32,
        n_trainable_layers=2,
        bilstm_hidden_size=256,
        device='cuda',
        p_list=(0.3, 0.3, 0.3),
        pre_classifier=True,
        classify=False
    ):

        super(AudioModel, self).__init__()
        self.chunk_size = chunk_length
        self.downsampler_out_dim = downsampler_out_dim
        self.device = device
        self.classify = classify
        self.__bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec2 = self.__bundle.get_model().feature_extractor
        
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        for layer in self.wav2vec2.conv_layers[-n_trainable_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.downsampler = nn.Linear(249, downsampler_out_dim)
        self.dropout1 = nn.Dropout(p=p_list[0])
        self.bilstm = nn.LSTM(input_size=512, hidden_size=bilstm_hidden_size, bidirectional=True)
        self.dropout2 = nn.Dropout(p=p_list[1])
        self.pre_classifier = pre_classifier
        if pre_classifier:
            self.pre_classifier = nn.Linear(512, 512)
            self.dropout3 = nn.Dropout(p=p_list[2])
        self.relu = nn.ReLU()
        if classify:
            self.classifier= nn.Linear(512, 1)
        
    
    def forward(self, waves):
        outs = []
        for wave in waves:
            wave = wave.to(self.device)
            x, _ = self.wav2vec2(wave, None)
            x = x.transpose(1, 2)
            x = self.downsampler(x)
            outs.append(x)
        x = torch.cat(outs, dim=2)
        x = x.transpose(1, 2)
        x = self.dropout1(x)
        x = self.bilstm(x)[0]
        x = x[:,-1,:]
        x = self.dropout2(x)
        if self.pre_classifier:
            x = self.pre_classifier(x)
            x = self.dropout3(x)
        x = self.relu(x)
        if self.classify:
            x = self.classifier(x)
        return x