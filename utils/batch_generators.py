import torch

def batch_generator_text(batch):
    """
        Generates the batch for the TextModel. It pads the sequences up to the lenght of the longest sequence in the batch and
        group them by their key.

        Parameters
        ----------
        batch: list of tuples
            Where each tuple contains (input_example, label), with input_example that is a dictionary of type 
            {'input_ids': [], 'attention_mask': []} and label that is the corresponding label id.
        
        Returns
        ----------
        results: tuple
            A tuple containing:
                - A dictionary containg, for each key, a batch tensor representing the input.
                - A tensor containing the labels of the batch.
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
            Where each tuple is of the type (input_example, label), with input_example which is already a tensor and 
            label which is the corresponding label id.

        Returns
        ----------
        results: tuple
            A tuple containing:
                - A tensor containing the input of the batch.
                - A tensor containing the labels of the batch.
    """
    return torch.stack([b[0] for b in batch], dim = 0), torch.FloatTensor([b[1] for b in batch])

def batch_generator_multimodal(batch):
    """
        Generates the batch for the MultimodalModel. Calls the batch generator functions for the audio and text model and joins the results.

        Parameters
        ----------
        batch: list of tuple
            Where each tuple is of the type (text_input, audio_input, label), with text_input a dictionary of the form {'input_ids': [], 'attention_mask': []},
            audio_input a tensor and label the corresponding label id
        
        Returns
        ----------
        results: tuple
            A tuple containing:
                - A tensor containing the text input of the batch.
                - A tensor containing the audio input of the batch.
                - A tensor containing the labels of the batch.
    """
    batch_text = [[b[0], b[2]] for b in batch]
    batch_audio = [[b[1], b[2]] for b in batch]
    text_tensor, _ = batch_generator_text(batch_text)
    audio_tensor, labels = batch_generator_wav2vec(batch_audio)
    return text_tensor, audio_tensor, labels

def batch_generator_mult_bart(batch):
    """
        Generates the batch for the TextModel. It pads the sequences up to the lenght of the longest sequence in the batch and
        group them by their key.
        Parameters
        ----------
        batch: list of tuples
            Where each tuple contains (input_example, label), with input_example that is a dictionary of type 
            {'input_ids': [], 'attention_mask': []} and label that is the corresponding label id.
        
        Returns
        ----------
        results: tuple
            A tuple containing:
                - A dictionary containg, for each key, a batch tensor representing the input.
                - A tensor containing the labels of the batch.
    """
    max_len = 0
    max_len_motion = 0

    # Get the maximum sequence lenght in the batch
    for data in batch:
        max_len = max(max_len, len(data[0]['input_ids']))
        max_len_motion = max(max_len_motion, len(data[2]))
    
    # Pad the sequences in the batch to max_len
    for data in batch:
        if len(data[0]['input_ids']) < max_len: 
            for key in data[0].keys():
                value = 1 if key == 'input_ids' else 0
                data[0][key] = torch.nn.functional.pad(data[0][key], (0, max_len - len(data[0][key])), value=value)
        if len(data[2]) < max_len_motion: 
                value = -100 # Set the padded tokens to -100 in labels
                data[2] = torch.nn.functional.pad(data[2], (0, max_len_motion - len(data[2])), value=value)
    
    # Group text sequences by key
    keys_text = batch[0][0].keys()
    texts = {key: torch.stack([data[0][key] for data in batch]) for key in keys_text}
    motions = torch.stack([data[2] for data in batch]) 

    audio, label_cls = batch_generator_wav2vec([[b[1], b[3]] for b in batch])
    
    return texts, audio, motions, label_cls