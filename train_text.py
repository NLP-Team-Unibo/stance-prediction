from torch import optim
from torch import nn
import torch

def train_loop(model, loader_train, loader_val, epochs, device):
    if device == 'cuda':
        model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()

    for i in range(epochs):
        train(model, optimizer, criterion, loader_train, device)
        validate(model, criterion, loader_val, device)

def train(model, optimizer, criterion, data_loader, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    for data in data_loader:
        #input_ids, segment_tensors, attention_mask = [torch.squeeze(d) for d in data[0].split(1, dim=1)]
        #input_ids, segment_tensors, attention_mask = input_ids.to(device), segment_tensors.to(device), attention_mask.to(device)

        input_dict = data[0]
        input_dict = {k:input_dict[k].to(device) for k in input_dict.keys()}
        
        labels = data[2].to(device)
        #output = model(input_ids, segment_tensors, attention_mask)
        output = model(**input_dict)
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

            labels = data[2].to(device)
            #output = model(input_ids, segment_tensors, attention_mask)
            output = model(**input_dict)
            output = output.squeeze(1)

            loss = criterion(output, labels)
            total_loss += loss.item()
            total += labels.size(0)

            acc = ((output > 0).float() == labels).sum().item()
            total_acc += acc

        print('val_loss:', total_loss / len(data_loader), 'val_accuracy:', total_acc / total)