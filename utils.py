import torch
from models.audio_model import AudioModel
from models.text_model import TextModel
from models.multimodal_model import MultimodalModel

# Data loader creation
# Expected batch: list of text features, audio features, stances with shape respectively of [batch_size, 3, max_len], [batch_size, audio_features_shape??], [batch_size] 
def batch_generator_text(batch):
    max_len = 0
    for data in batch:
        max_len = max(max_len, len(data[0]['input_ids'])) ##change
    
    for data in batch:
        if len(data[0]['input_ids']) < max_len:
            for key in data[0].keys():
                data[0][key] = torch.nn.functional.pad(data[0][key], (0, max_len - len(data[0][key])))

    keys = batch[0][0].keys()
    texts = {}
    for key in keys:
        tensors = []
        for data in batch:
            tensors.append(data[0][key])
        texts[key] = torch.stack(tensors, dim=0)
    

    return texts, torch.FloatTensor([b[1] for b in batch])

def batch_generator_wav2vec(batch):
    """max_chunks = 10000
    for b in batch:
        max_chunks = min(len(b[0]), max_chunks)
    for i in range(len(batch)):
        batch[i][0] = batch[i][0][:max_chunks]
    x = torch.stack([torch.stack(b[0], dim=0) for b in batch], dim=0)
    x = x.transpose(0, 1)
    x = torch.split(x, 1)
    x = [x.squeeze(0) for x in x]"""
    return torch.stack([b[0] for b in batch], dim = 0), torch.FloatTensor([b[1] for b in batch])

def batch_generator_multimodal(batch):
    batch_text = [[b[0], b[2]] for b in batch]
    batch_audio = [[b[1], b[2]] for b in batch]
    text_tensor, _ = batch_generator_text(batch_text)
    audio_tensor, labels = batch_generator_wav2vec(batch_audio)
    return text_tensor, audio_tensor, labels

def get_params_groups(model, optimizer_args):
    params = []
    for name, module in model.named_modules():
        for i in range(len(optimizer_args['params'])):
            if optimizer_args['params'][i] == name:
                params.append({'params':module.parameters(), 'lr':optimizer_args['lr'][i]})
    return params

def get_model(cfg):
    model = None
    model_name = cfg.MODEL.NAME
    models = []
    if model_name == 'text' or model_name == 'multimodal':
        models.append(TextModel(
                            distilbert_type=cfg.MODEL.TEXT.DISTILBERT,
                            n_trainable_layers=cfg.MODEL.TEXT.N_TRAINABLE_LAYERS,
                            p_list=cfg.MODEL.TEXT.DROPOUT_VALUES,
                            pre_classifier=cfg.MODEL.TEXT.PRE_CLASSIFIER,
                            classify=cfg.MODEL.TEXT.CLASSIFY
                        )
                    )
    if model_name == 'audio' or model_name == 'multimodal':
        models.append(AudioModel(
                            chunk_length=cfg.DATASET.CHUNK_LENGTH, 
                            n_transformers=cfg.MODEL.AUDIO.N_TRANSFORMERS,
                            n_trainable_layers=cfg.MODEL.AUDIO.N_TRAINABLE_LAYERS,
                            p_list=cfg.MODEL.AUDIO.DROPOUT_VALUES,
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
                        p_list=cfg.MODEL.MULTIMODAL.DROPOUT_VALUES,
                        freeze_text=cfg.MODEL.MULTIMODAL.FREEZE_TEXT,
                        freeze_audio=cfg.MODEL.MULTIMODAL.FREEZE_AUDIO
                    )
    else:
        model = models[0]
    return model
