import os

import torch
from torch import nn

class StancePredictionModule(nn.Module):
    def __init__(self):
        """
            Subclass of nn.Module. If inherited, allows to save and load the state_dict of a stance prediction module.
        """
        super(StancePredictionModule, self).__init__()
    
    def save_backbone(self, path):
        """
            Save the state_dict of this object to the specified path.
    
            Parameters
            ----------
            path: str
                Output destination path.            
        """
        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_backbone(self, path, drop_classifier=False):
        """
            Load the state_dict of this object from the specified path, allowing also to decide to drop or not the classifier.
    
            Parameters
            ----------
            path: str
                Output destination path.
            drop_classifier: bool
                If True, drop the classifier's weights from the state_dict, else keep everything. Default False.
        """
        state_dict = torch.load(path)
        if drop_classifier:
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
        self.load_state_dict(torch.load(path), strict=False)
