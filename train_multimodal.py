from torch import optim
from torch import nn
import torch
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
def train_loop(model, loader_train, loader_val, epochs, device):
    if device == 'cuda':
        model.cuda()
    writer = SummaryWriter('log/', flush_secs=1)
    optimizer = optim.Adam([
        {'params': model.audio_model.parameters(), 'lr':2e-5},
        {'params': model.classifier.parameters(), 'lr':2e-5},
        {'params': model.text_model.parameters(), 'lr':2e-5},
        ])
    criterion = nn.BCEWithLogitsLoss()
    for i in range(epochs):
        train(model, optimizer, criterion, loader_train, device, writer, i)
        validate(model, criterion, loader_val, device)

def train(model, optimizer, criterion, data_loader, device, writer, epoch):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    for i, data in enumerate(tqdm(data_loader)):
        input_dict = data[0]
        input_dict = {k:input_dict[k].to(device) for k in input_dict.keys()}
        waves = data[1]
        labels = data[2].to(device)
        output = model(input_dict, waves)
        output = output.squeeze(1)
        loss = criterion(output, labels)
        total_loss += loss.item()
        writer.add_scalar('Batch Loss', loss.item(), i)
        acc = ((output > 0).float() == labels).sum().item()
        total_acc += acc
        total += labels.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    writer.add_scalar('Train Loss', total_loss / len(data_loader), epoch)
    writer.add_scalar('Train Accuracy', total_acc / total, epoch)
    print('train_loss:', total_loss / len(data_loader), 'train_accuracy:', total_acc / total, end='\t')

def validate(model, criterion, data_loader, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_acc = 0.0
        total = 0
        for data in data_loader:
            #input_ids, segment_tensors, attention_mask = [torch.squeeze(d) for d in data[0].split(1, dim=1)]
            #input_ids, segment_tensors, attention_mask = input_ids.to(device), segment_tensors.to(device), attention_mask.to(device)
            input_dict = data[0]
            input_dict = {k:input_dict[k].to(device) for k in input_dict.keys()}

            waves = data[1]
            labels = data[2].to(device)
            #output = model(input_ids, segment_tensors, attention_mask)
            output = model(input_dict, waves)
            output = output.squeeze(1)

            loss = criterion(output, labels)
            total_loss += loss.item()
            total += labels.size(0)

            acc = ((output > 0).float() == labels).sum().item()
            total_acc += acc

        print('val_loss:', total_loss / len(data_loader), 'val_accuracy:', total_acc / total)