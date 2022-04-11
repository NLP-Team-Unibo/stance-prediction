import torch
from torch import nn
import torchaudio

class AudioModel(nn.Module):
    def __init__(self, chunk_size=10, dim=32, device='cuda', classify=False):
        super(AudioModel, self).__init__()
        self.chunk_size = chunk_size
        self.dim = dim
        self.device = device
        self.classify = classify
        self.__bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec2 = self.__bundle.get_model().feature_extractor
        
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        for layer in self.wav2vec2.conv_layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.linear = nn.Linear(249, dim)
        self.classifier= nn.Linear(512, 1)
        self.relu = nn.ReLU()
    
    def forward(self, waves):
        outs = []
        for wave in waves:
            wave = wave.to(self.device)
            x, _ = self.wav2vec2(wave, None)
            x = x.transpose(1, 2)
            x = self.linear(x)
            outs.append(x)
        
        x = torch.cat(outs, dim=2)
        x = torch.mean(x, dim=2)
        if self.classify:
            x = self.classifier(x)
            x = self.relu(x)
        return x