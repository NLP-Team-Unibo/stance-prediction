import torch

def batch_generator_text(batch):
    """
        Generates the batch for the TextModel. It pads the sequences up to the lenght of the longest sequence in the batch and
        group them by their key.

        Parameters
        ----------
        batch: list of tuples
            where each tuple contains (input_example, label), with input_example that is a dictionary of type 
            {'input_ids': [], 'attention_mask': []} and label that is the corresponding label id.
    """
    max_len = 0

    # Get the maximum sequence lenght in the batch
    for data in batch:
        max_len = max(max_len, len(data[0]['input_ids']))
    
    # Pad the sequences in the batch to max_len
    for data in batch:
        if len(data[0]['input_ids']) < max_len: 
            for key in data[0].keys():
                data[0][key] = torch.nn.functional.pad(data[0][key], (0, max_len - len(data[0][key])))
    
    # Group text sequences by key
    keys = batch[0][0].keys()
    texts = {key: torch.stack([data[0][key] for data in batch]) for key in keys}
    
    return texts, torch.FloatTensor([b[1] for b in batch])

def batch_generator_wav2vec(batch):
    """
        Generates the batch for the AudioModel. Groups the input samples in a tensor.

        Parameters
        ----------
        batch: list of tuple
            where each tuple is of the type (input_example, label), with input_example which is already a tensor and 
            label which is the corresponding label id.
    """
    return torch.stack([b[0] for b in batch], dim = 0), torch.FloatTensor([b[1] for b in batch])

def batch_generator_multimodal(batch):
    """
        Generates the batch for the MultimodalModel. Calls the batch generator functions for the audio and text model and joins the results.

        Parameters
        ----------
        batch: list of tuple
            where each tuple is of the type (text_input, audio_input, label), with text_input a dictionary of the form {'input_ids': [], 'attention_mask': []},
            audio_input a tensor and label the corresponding label id
    """
    batch_text = [[b[0], b[2]] for b in batch]
    batch_audio = [[b[1], b[2]] for b in batch]
    text_tensor, _ = batch_generator_text(batch_text)
    audio_tensor, labels = batch_generator_wav2vec(batch_audio)
    return text_tensor, audio_tensor, labels

