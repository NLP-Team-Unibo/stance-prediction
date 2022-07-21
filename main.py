import numpy as np
from argparse import ArgumentParser
from torchinfo import summary

import torch
import torchtext
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

import evaluate
import transformers
from transformers import AutoTokenizer

from config import config
from train import train_loop
from ibm_dataset import IBMDebater

from utils.train import *
from utils.early_stopping import *
from utils.batch_generators import *

transformers.logging.set_verbosity_error()


def train_pipeline(args):
    """
        This function excecute the training pipeline according to the configuration file. In particular it excecutes the following tasks:
            - pre-processing: defines how raw data will be transformed in order to make it suitable for training the model;
            - data splitting: splits the dataset into train and validation;
            - data loading: splits the data in batches;
            - model definition: define the model according to the configuration file.
            - training: excecute the training procedure.
    """
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
    tokenizer = AutoTokenizer.from_pretrained(cfg.DATASET.TOKENIZER)
    sample_cut_type = cfg.DATASET.SAMPLE_CUT_TYPE
    load_motion = cfg.DATASET.LOAD_MOTION

    # Define how the data will be pre-processed by calling IBMDebater
    data_train = IBMDebater(data_path, 
                    split='train', 
                    tokenizer=tokenizer, 
                    chunk_length=chunk_length, 
                    text_transform=text_transform,
                    load_audio=load_audio,
                    load_text=load_text,
                    sample_cut_type=sample_cut_type,
                    load_motion=load_motion)
    data_val = IBMDebater(data_path, 
                    split='validation', 
                    tokenizer=tokenizer, 
                    chunk_length=chunk_length, 
                    text_transform=text_transform,
                    load_audio=load_audio,
                    load_text=load_text,
                    sample_cut_type=sample_cut_type,
                    load_motion=load_motion)

    # Splits the whole dataset into train and validation.
    # If specified, use just a small subset of the original dataset.
    if cfg.DATASET.SMALL_VERSION:
        small_data_dim = 0.2
        rnd_idx = np.random.choice(np.array([i for i in range(1, len(data_train))]), size=int(len(data_train)*small_data_dim))
        data_train = torch.utils.data.Subset(data_train, rnd_idx)

        rnd_idx = np.random.choice(np.array([i for i in range(1, len(data_val))]), size=int(len(data_val)*small_data_dim))
        data_val = torch.utils.data.Subset(data_val, rnd_idx)

    # Specify the batch collate function according to the type of model
    if model_name == 'text':
        collate_fn = batch_generator_text
    elif model_name == 'audio':
        collate_fn = batch_generator_wav2vec
    elif model_name == 'text_generation':
        collate_fn = batch_generator_mult_bart
    else:
        collate_fn = batch_generator_multimodal

    # Data loading task: prepare the data loaders, in order to split the data in batches
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
                        drop_last=False,
                        num_workers=num_workers)

    # Get the model accoriding to the configuration file
    model = get_model(cfg)
    summary(model)

    # Set up optimizer, scheduler and other training loop parameters/utils according to the configuration file
    optimizer = cfg.TRAIN.OPTIMIZER
    optimizer_args = cfg.TRAIN.OPTIMIZER_ARGS
    scheduler = cfg.TRAIN.LR_SCHEDULER
    early_stopping = cfg.TRAIN.EARLY_STOPPING
    lr = cfg.TRAIN.LR
    epochs = cfg.TRAIN.EPOCHS
    params = [{'params': model.parameters(), 'lr':lr}]
    if len(optimizer_args) > 0:
        params = get_params_groups(model, optimizer_args)
    if optimizer == 'adam':
        optimizer = optim.Adam(params, lr=lr)
    if len(scheduler) > 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler)
    early_stopping = EarlyStopping(model, patience=early_stopping.PATIENCE)
    criterion = nn.BCEWithLogitsLoss()

    gen_metrics = None
    if len(cfg.TRAIN.GENERATION_METRICS) > 0 and model_name == 'text_generation':
        gen_metrics =cfg.TRAIN.GENERATION_METRICS
    
    # Start train loop and save checkpoints at the end if the configuration file specifies it
    train_loop(model, optimizer, criterion, early_stopping, loader_train, loader_val, epochs, device, step_lr=scheduler, cfg=cfg, gen_metrics=gen_metrics, tokenizer=tokenizer)
    if cfg.TRAIN.SAVE_CHECKPOINT:
        path = cfg.TRAIN.CHECKPOINT_PATH
        model.save_backbone(path)

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('cfg_path', help='Path of the model\'s configuration file')
    args = args.parse_args()
    train_pipeline(args)
