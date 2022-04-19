from torch import optim
from torch import nn
import torch
from tqdm import tqdm
from early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
def train_loop(
        model, 
        optimizer, 
        criterion, 
        early_stopping, 
        loader_train, 
        loader_val, 
        epochs, 
        device,
        step_lr=None,
        cfg=''
    ):
    if device == 'cuda':
        model.cuda()
    writer = SummaryWriter('log/', flush_secs=1)
    writer.add_text('CFG', text_string=str(cfg))

    for i in range(epochs):
        train_results = train(model, optimizer, criterion, loader_train, device)
        val_results = validate(model, criterion, loader_val, device)

        writer.add_scalar('Train loss', train_results['train_loss'], i)
        writer.add_scalar('Train accuracy', train_results['train_accuracy'], i)
        writer.add_scalar('Val loss', val_results['val_loss'], i)
        writer.add_scalar('Val accuracy', val_results['val_accuracy'], i)
        if early_stopping:
            if early_stopping(val_results['val_accuracy']):
                model.load_state_dict(early_stopping.best_weights)
                break
        if step_lr:
            step_lr.step()
        

def train(model, optimizer, criterion, data_loader, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    results = {}
    model_name = model.__class__.__name__
    for data in tqdm(data_loader):
        if model_name == 'TextModel':
            input_dict = data[0]
            input_dict = {k:input_dict[k].to(device) for k in input_dict.keys()}
            labels = data[1].to(device)
            output = model(**input_dict)
        elif model_name == 'AudioModel':
            waves = data[0]
            labels = data[1].to(device)
            output = model(waves)
        else:
            input_dict = data[0]
            input_dict = {k:input_dict[k].to(device) for k in input_dict.keys()}
            waves = data[1]
            labels = data[2].to(device)
            output = model(input_dict, waves)
        output = output.squeeze(1)
        loss = criterion(output, labels)
        total_loss += loss.item()

        acc = ((output > 0).float() == labels).sum().item()
        total_acc += acc
        total += labels.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print('train_loss:', total_loss / len(data_loader), 'train_accuracy:', total_acc / total, end='\t')
    results['train_loss'] = total_loss / len(data_loader)
    results['train_accuracy'] = total_acc / total
    return results

def validate(model, criterion, data_loader, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_acc = 0.0
        total = 0
        results = {}
        model_name = model.__class__.__name__
        for data in data_loader:
            if model_name == 'TextModel':
                input_dict = data[0]
                input_dict = {k:input_dict[k].to(device) for k in input_dict.keys()}
                labels = data[1].to(device)
                output = model(**input_dict)
            elif model_name == 'AudioModel':
                waves = data[0]
                labels = data[1].to(device)
                output = model(waves)
            else:
                input_dict = data[0]
                input_dict = {k:input_dict[k].to(device) for k in input_dict.keys()}
                waves = data[1]
                labels = data[2].to(device)
                output = model(input_dict, waves)
            output = output.squeeze(1)

            loss = criterion(output, labels)
            total_loss += loss.item()
            total += labels.size(0)

            acc = ((output > 0).float() == labels).sum().item()
            total_acc += acc
        results['val_loss'] = total_loss / len(data_loader)
        results['val_accuracy'] = total_acc / total
        print('val_loss:', total_loss / len(data_loader), 'val_accuracy:', total_acc / total)
    return results