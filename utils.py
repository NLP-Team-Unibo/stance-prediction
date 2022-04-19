import torch

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
    max_chunks = 10000
    for b in batch:
        max_chunks = min(len(b[0]), max_chunks)
    for i in range(len(batch)):
        batch[i][0] = batch[i][0][:max_chunks]
    x = torch.stack([torch.stack(b[0], dim=0) for b in batch], dim=0)
    x = x.transpose(0, 1)
    x = torch.split(x, 1)
    x = [x.squeeze(0) for x in x]
    return x, torch.FloatTensor([b[1] for b in batch])

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