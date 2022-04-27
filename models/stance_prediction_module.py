
import torch
from torch import nn
import os

class StancePredictionModule(nn.Module):

    def __init__(self):
        super(StancePredictionModule, self).__init__()
    
    def save_backbone(self, path):
        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_backbone(self, path, drop_classifier=False):
        state_dict = torch.load(path)
        if drop_classifier:
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
        self.load_state_dict(torch.load(path), strict=False)
