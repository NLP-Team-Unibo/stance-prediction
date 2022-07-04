import torch
import torchaudio
from torch import nn

from models.stance_prediction_module import StancePredictionModule

class AudioModel(StancePredictionModule):
    def __init__(
        self, 
        n_transformers=12,
        n_trainable_layers=2,
        dropout_values=(0.3, 0.3),
        pre_classifier=True,
        classify=False,
        return_sequences=False
    ):
        """
            Creates the model for the speech classification task. It relies on Wav2Vec2-BASE for the feature extraction and encoding part of the raw signal, 
            then one or two classification heads are attached to the temporal average of the output of the encoder.
    
            Parameters
            ----------
            n_transformers: int
                Number of transformers to keep in the encoding part of Wav2Vec2, starting from the first one. Default to 12.
            n_trainable_layers: int
                Number of transformer layers to fine-tune. All the layers before are fixed to pre-trained weights. Default to 2.
            dropout_values: tuple of floats
                The values to be used for dropout, respectively before and after the pre-classifier part of the network. Default to (0.3, 0.3)
            pre_classifier: bool
                Whether to add or not another linear layer before the final one. Default to True.
            classify: bool
                Whether to add or not the final layer, which takes care of making the prediction. This will typically be set to
                False only when the AudioModel is used as a component in the bigger MultimodalModel. Default to False.
            
        """

        super(AudioModel, self).__init__()
        assert n_transformers >= n_trainable_layers, 'Number of transformer layers must be greater or equal to the number of trainable layers'
        assert 1 <= n_transformers <= 12, 'Number of transformer layers must be between 1 and 12'
        self.classify = classify
        self.__bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec2 = self.__bundle.get_model()
        self.return_sequences = return_sequences

        # Keeping only the first n_transformers
        self.wav2vec2.encoder.transformer.layers = self.wav2vec2.encoder.transformer.layers[:n_transformers]

        self.wav2vec2_out_dim = self.wav2vec2.encoder.transformer.layers[-1].final_layer_norm.normalized_shape[0]
        
        # Freezing all the parameters but the last 'n_trainable_layers'
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        if n_trainable_layers > 0:
            for layer in self.wav2vec2.encoder.transformer.layers[-n_trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        if not self.return_sequences:
            # Defining the classification heads
            self.dropout1 = nn.Dropout(p=dropout_values[0])
            self.pre_classifier = pre_classifier
            if pre_classifier:
                self.pre_classifier = nn.Linear(self.wav2vec2_out_dim, self.wav2vec2_out_dim)
                self.dropout2 = nn.Dropout(p=dropout_values[1])
            self.relu = nn.ReLU()
            if classify:
                self.classifier= nn.Linear(self.wav2vec2_out_dim, 1)
        
    
    def forward(self, audio):
        # Feature extraction + encoding
        x, _ = self.wav2vec2(audio)
        if self.return_sequences:
            return x
        # Temporal average
        x = torch.mean(x, dim=1)

        # Classification
        x = self.dropout1(x)
        if self.pre_classifier:
            x = self.pre_classifier(x)
            x = self.dropout2(x)
        x = self.relu(x)
        if self.classify:
            x = self.classifier(x)
        return x