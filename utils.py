import torch

# Data loader creation
# Expected batch: list of text features, audio features, stances with shape respectively of [batch_size, 3, max_len], [batch_size, audio_features_shape??], [batch_size] 
def batch_generator(batch):
    max_len = 0
    for data in batch:
        ids = data[0][0]
        max_len = max(max_len, len(ids))
    
    for data in batch:
        if len(data[0][0]) < max_len:
            data[0][0] = torch.nn.functional.pad(data[0][0], (0, max_len - len(data[0][0])))
            data[0][1] = torch.nn.functional.pad(data[0][1], (0, max_len - len(data[0][1])))
            data[0][2] = torch.nn.functional.pad(data[0][2], (0, max_len - len(data[0][2])))
    
    b = [b[0] for b in batch]
    b = [torch.stack(b, dim=0) for b in b]
    return torch.stack(b, dim=0), [b[1] for b in batch], torch.FloatTensor([b[2] for b in batch])