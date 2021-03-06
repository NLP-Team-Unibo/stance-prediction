import os
from tqdm import tqdm
from argparse import ArgumentParser
import evaluate

import torch
import torchtext
from torchinfo import summary
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer

from config import config
from ibm_dataset import IBMDebater

from utils.train import *
from utils.early_stopping import *
from utils.batch_generators import *
from utils.train import get_decoded_preds_and_labels
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

transformers.logging.set_verbosity_error()

def evaluate_pipeline(args):
    """
        This function excecute the evaluation pipeline according to the configuration file. In particular it excecutes the following tasks:
            - pre-processing: defines how raw data will be transformed in order to make it suitable for evaluate the model;
            - data loading: splits the data in batches;
            - model loading: define and load the model checkpoint according to the configuration file.
            - evaluating: excecute the evaluation procedure.
    """
    checkpoint_path = args.checkpoint_path
    cfg_path = args.cfg_path
    device = args.device
    cfg = config.get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()


    data_path = cfg.DATASET.DATA_PATH
    load_audio = cfg.DATASET.LOAD_AUDIO
    load_text = cfg.DATASET.LOAD_TEXT
    chunk_length = cfg.DATASET.CHUNK_LENGTH
    text_transform = torchtext.transforms.ToTensor()
    tokenizer = AutoTokenizer.from_pretrained(cfg.DATASET.TOKENIZER)
    sample_cut_type = cfg.DATASET.SAMPLE_CUT_TYPE
    load_motion = cfg.DATASET.LOAD_MOTION

    # Define how the data will be pre-processed by calling IBMDebater
    data_test = IBMDebater(data_path, 
                    split='test', 
                    tokenizer=tokenizer, 
                    chunk_length=chunk_length, 
                    text_transform=text_transform,
                    load_audio=load_audio,
                    load_text=load_text,
                    sample_cut_type=sample_cut_type,
                    load_motion=load_motion)
    
    model_name = cfg.MODEL.NAME

    # Specify the batch collate function according to the type of model
    if model_name == 'text':
        collate_fn = batch_generator_text
    elif model_name == 'audio':
        collate_fn = batch_generator_wav2vec
    elif model_name == 'text_generation':
        collate_fn = batch_generator_mult_bart if load_audio else batch_generator_bart
    else:
        collate_fn = batch_generator_multimodal

    # Data loading task: prepare the data loader, in order to split the data in batches
    batch_size = cfg.DATASET.LOADER.BATCH_SIZE
    num_workers = cfg.DATASET.LOADER.NUM_WORKERS
    loader_test = DataLoader(data_test,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_fn,
                        drop_last=False,
                        num_workers=num_workers)

    gen_metrics = None
    if len(cfg.TRAIN.GENERATION_METRICS) > 0 and model_name == 'text_generation':
        gen_metrics = cfg.TRAIN.GENERATION_METRICS

    # Model loading: define the model and load its checkpoint, according to the cfg file
    state_dict = torch.load(checkpoint_path, device)
    model = get_model(cfg)
    model.load_state_dict(state_dict)
    if device == 'cuda':
        model.cuda()
    summary(model)
	
    # Starts the evaluation procedure
    y_pred, y_true = test(model, loader_test, device, gen_metrics=gen_metrics, tokenizer=tokenizer)
    y_pred, y_true = y_pred.cpu().numpy(), y_true.cpu().numpy()
    
    mask = ~(y_pred == y_true)
    os.makedirs('wrongs/' + model_name, exist_ok=True)
    data_test.annotations['speech-id'][mask].to_csv(f'wrongs/{model_name}/{cfg_path.split("/")[-1].replace(".yaml", ".csv")}')
    
    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['con', 'pro'], ax=ax, cmap=plt.cm.Blues, normalize='true')
    print(classification_report(y_true, y_pred, target_names=['con', 'pro']))
    os.makedirs('images/'+ model_name, exist_ok=True)
    plt.savefig(f'images/{model_name}/{cfg_path.split("/")[-1].replace(".yaml", ".png")}')

def test(model, data_loader, device, gen_metrics=None, tokenizer=None):
    """
        This function excecute a single evaluation step.

        Parameters
        ----------
        model: models.StancePredictionModule
            The stance prediction model to be evaluate.
        data_loader: torch.utils.data.DataLoader
            The data loader of the test dataset.
        device: str
            The name of the device where to excecute the evaluation procedure.       
        Returns
        ----------
        results: dict
    """
    model.eval()
    with torch.no_grad():
        gen_metrics = [evaluate.load(metric_name) for metric_name in gen_metrics]
        total_acc = 0.0
        total = 0
        model_name = model.__class__.__name__

        y_pred = []
        y_true = []
        for data in tqdm(data_loader):
            if model_name == 'TextModel':
                input_dict = data[0]
                input_dict = {k:input_dict[k].to(device) for k in input_dict.keys()}
                labels = data[1].to(device)
                output = model(**input_dict)
            elif model_name == 'AudioModel':
                waves = data[0].to(device)
                labels = data[1].to(device)
                output = model(waves)
            elif model_name == 'MultimodalModel':
                input_dict = data[0]
                input_dict = {k:input_dict[k].to(device) for k in input_dict.keys()}
                waves = data[1].to(device)
                labels = data[2].to(device)
                output = model(input_dict, waves)
            elif model_name == 'TextGenerationModel':
                input_dict = data['text']
                input_dict = {k:input_dict[k].to(device) for k in input_dict.keys()}
                waves = None
                motion = None
                if model.use_audio:
                    waves = data['audio'].to(device)
                if model.generate_motion:
                    motion = data['motion'].to(device)
                labels = data['labels'].to(device)
                
                _, _, output = model(input_dict['input_ids'], input_dict['attention_mask'], waves, motion, labels_cls=labels, return_dict=False)
            
            if model_name != 'TextGenerationModel':
                output = output.squeeze()

            total += labels.size(0)
            
            pred = (output > 0).float()
            y_pred.append(pred)
            y_true.append(labels)
            acc = (pred == labels).sum().item()
            total_acc += acc

            if model.generate_motion:
                # Compute text generation metrics
                if gen_metrics is not None:
                    preds, motions = get_decoded_preds_and_labels(**input_dict, audio=waves, labels=motion, model=model, tokenizer=tokenizer)
                    with open('gen_dump_evaluate.txt', 'a') as f:
                        f.write('\n' + '\n'.join(preds) + '\n')
                    for metric in gen_metrics:
                        metric.add_batch(predictions=preds, references=motions)

        if model.generate_motion:
            results = {}
            for metric in gen_metrics:
                try:
                    results.update(metric.compute())
                except ZeroDivisionError:
                    results[metric.__class__.__name__] = 0.0
            print(results)

        print('test_accuracy:', total_acc / total)
    return torch.cat(y_pred), torch.cat(y_true)

    

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('checkpoint_path', help='Path of the checkpoint file')
    args.add_argument('cfg_path', help='Path of the model\'s configuration file')
    args.add_argument('--device', '-d', default='cuda', help='Device name, default is \"cuda\"')
    args = args.parse_args()
    evaluate_pipeline(args)
