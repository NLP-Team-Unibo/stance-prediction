from models.audio_model import AudioModel
from models.text_model import TextModel
from models.multimodal_model import MultimodalModel

def get_params_groups(model, optimizer_args):
    """
        Assigns at each sub-module of the model the initial learning rate defined in 'optimizer_args'.

        Parameters
        ----------
        model: nn.Module
        optimizer_args: dict
    """
    params = []
    for name, module in model.named_modules():
        for i in range(len(optimizer_args['params'])):
            if optimizer_args['params'][i] == name:
                params.append({'params':module.parameters(), 'lr':optimizer_args['lr'][i]})
    return params

def get_model(cfg):
    """
        Creates and returns the model according to the configuration file

        Parameters
        ----------
        cfg: yacs.config.CfgNode

        Returns
        -------
        model: nn.Module
            pytorch model created according to the config file in cfg
    
    """
    model = None
    model_name = cfg.MODEL.NAME
    models = []
    if model_name == 'text' or model_name == 'multimodal':
        models.append(TextModel(
                            distilbert_type=cfg.MODEL.TEXT.DISTILBERT,
                            n_trainable_layers=cfg.MODEL.TEXT.N_TRAINABLE_LAYERS,
                            dropout_values=cfg.MODEL.TEXT.DROPOUT_VALUES,
                            pre_classifier=cfg.MODEL.TEXT.PRE_CLASSIFIER,
                            classify=cfg.MODEL.TEXT.CLASSIFY
                        )
                    )
    if model_name == 'audio' or model_name == 'multimodal':
        models.append(AudioModel(
                            chunk_length=cfg.DATASET.CHUNK_LENGTH, 
                            n_transformers=cfg.MODEL.AUDIO.N_TRANSFORMERS,
                            n_trainable_layers=cfg.MODEL.AUDIO.N_TRAINABLE_LAYERS,
                            dropout_values=cfg.MODEL.AUDIO.DROPOUT_VALUES,
                            pre_classifier=cfg.MODEL.AUDIO.PRE_CLASSIFIER,
                            classify=cfg.MODEL.AUDIO.CLASSIFY
                        )
                    )
    if cfg.MODEL.NAME == 'multimodal':
        if cfg.MODEL.MULTIMODAL.LOAD_TEXT_CHECKPOINT:
            models[0].load_backbone(cfg.MODEL.MULTIMODAL.TEXT_CHECKPOINT_PATH, drop_classifier=True)
        if cfg.MODEL.MULTIMODAL.LOAD_AUDIO_CHECKPOINT:
            models[1].load_backbone(cfg.MODEL.MULTIMODAL.AUDIO_CHECKPOINT_PATH, drop_classifier=True)
        model = MultimodalModel(
                        text_model=models[0],
                        audio_model=models[1],
                        dropout_values=cfg.MODEL.MULTIMODAL.DROPOUT_VALUES,
                        freeze_text=cfg.MODEL.MULTIMODAL.FREEZE_TEXT,
                        freeze_audio=cfg.MODEL.MULTIMODAL.FREEZE_AUDIO
                    )
    else:
        model = models[0]
    return model