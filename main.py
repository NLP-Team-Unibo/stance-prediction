import torchtext
from torch.utils.data import DataLoader, random_split
import transformers
from transformers import DistilBertTokenizer
from ibm_dataset import IBMDebater
import utils
from train_text import train_loop
from models.text_model import TextModel
from models.audio_model import AudioModel
from models.multimodal_model import MultimodalModel
from transformers import DistilBertTokenizer
import torch
import sys
import numpy as np
from torch import nn
from torch import optim
from early_stopping import EarlyStopping
from train import train_loop
transformers.logging.set_verbosity_error()
from config import config

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
                            downsampler_out_dim=cfg.MODEL.AUDIO.DOWNSAMPLER_OUT_DIM,
                            n_trainable_layers=cfg.MODEL.AUDIO.N_TRAINABLE_LAYERS,
                            bilstm_hidden_size=cfg.MODEL.AUDIO.BILSTM_HIDDEN_SIZE,
                            device=cfg.SETTINGS.DEVICE,
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


def train_pipeline(cfg):
    model_name = cfg.MODEL.NAME
    device = cfg.SETTINGS.DEVICE
    data_path = cfg.DATASET.DATA_PATH
    load_audio = cfg.DATASET.LOAD_AUDIO
    load_text = cfg.DATASET.LOAD_TEXT
    chunk_length = cfg.DATASET.CHUNK_LENGTH
    text_transform = torchtext.transforms.ToTensor()
    tokenizer = DistilBertTokenizer.from_pretrained(cfg.DATASET.TOKENIZER)


    data = IBMDebater(data_path, 
                    split='train', 
                    tokenizer=tokenizer, 
                    max_audio_len=chunk_length, 
                    text_transform=text_transform,
                    load_audio=load_audio,
                    load_text=load_text)

    train_len = int(len(data)*0.7)

    if cfg.DATASET.SMALL_VERSION:
        small_data_dim = 0.2
        rnd_idx = np.random.choice(np.array([i for i in range(1, len(data))]), size=int(len(data)*small_data_dim))
        small_data = torch.utils.data.Subset(data, rnd_idx)
        train_len = int(len(small_data)*0.7) 
        data_train, data_val = random_split(small_data, [train_len, len(small_data) - train_len])
    else:
        data_train, data_val = random_split(data, [train_len, len(data) - train_len])

    if model_name == 'text':
        collate_fn = utils.batch_generator_text
    elif model_name == 'audio':
        collate_fn = utils.batch_generator_wav2vec
    else:
        collate_fn = utils.batch_generator_multimodal

    batch_size = cfg.DATASET.LOADER.BATCH_SIZE
    drop_last = cfg.DATASET.LOADER.DROP_LAST
    loader_train = DataLoader(data_train,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=collate_fn,
                        drop_last=drop_last)
    loader_val = DataLoader(data_val,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_fn,
                        drop_last=drop_last)

    model = get_model(cfg)

    optimizer = cfg.TRAIN.OPTIMIZER
    optimizer_args = cfg.TRAIN.OPTIMIZER_ARGS
    scheduler = cfg.TRAIN.LR_SCHEDULER
    early_stopping = cfg.TRAIN.EARLY_STOPPING
    lr = cfg.TRAIN.LR
    epochs = cfg.TRAIN.EPOCHS
    params = [{'params': model.parameters(), 'lr':lr}]
    if len(optimizer_args) > 0:
        params = utils.get_params_groups(model, optimizer_args)
    if optimizer == 'adam':
        optimizer = optim.Adam(params, lr=lr)
    if len(scheduler) > 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler)
    early_stopping = EarlyStopping(model, patience=early_stopping.PATIENCE)
    criterion = nn.BCEWithLogitsLoss()

    train_loop(model, optimizer, criterion, early_stopping, loader_train, loader_val, epochs, device, step_lr=scheduler, cfg=cfg)

    if cfg.TRAIN.SAVE_CHECKPOINT:
        path = cfg.TRAIN.CHECKPOINT_PATH
        model.save_backbone(path)

if __name__ == '__main__':
    cfg = config.get_cfg_defaults()
    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
        cfg.merge_from_file(cfg_file)
    cfg.freeze()
    train_pipeline(cfg)
