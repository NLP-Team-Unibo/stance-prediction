from argparse import ArgumentParser
import torchtext
from torch.utils.data import DataLoader, random_split
import transformers
from transformers import DistilBertTokenizer
from ibm_dataset import IBMDebater
import utils
from train_text import train_loop
from transformers import DistilBertTokenizer
import torch
import sys
import numpy as np
from torch import nn, optim
from early_stopping import EarlyStopping
from train import train_loop
from config import config
transformers.logging.set_verbosity_error()


def train_pipeline(args):
    cfg_path = args.cfg_path

    cfg = config.get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

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
    num_workers = cfg.DATASET.LOADER.NUM_WORKERS
    loader_train = DataLoader(data_train,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=collate_fn,
                        drop_last=drop_last,
                        num_workers=num_workers)
    loader_val = DataLoader(data_val,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_fn,
                        drop_last=drop_last,
                        num_workers=num_workers)

    model = utils.get_model(cfg)

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
    args = ArgumentParser()
    args.add_argument('cfg_path', help='Path of the model\'s configuration file')
    args = args.parse_args()
    train_pipeline(args)
