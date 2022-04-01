import torch

# Data loader creation
# Expected batch: list of text features, audio features, stances with shape respectively of [batch_size, 3, max_len], [batch_size, audio_features_shape??], [batch_size] 
def batch_generator_bert(batch):
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
    return texts, [b[1] for b in batch], torch.FloatTensor([b[2] for b in batch])
#dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])