from torch.utils.data import Dataset
import os
import pandas as pd

class IBMDebater(Dataset):
    def __init__(self, path, split, tokenizer, text_transform=None, audio_transform=None):
        self.path = path
        self.tokenizer = tokenizer
        self.text_transform = text_transform
        self.audio_transform = audio_transform
        metadata_path = os.path.join(path, 'RecordedDebatingDataset_Release5_metadata.csv')
        split_path = os.path.join(path, 'OutOfTheEchoChamber_acl2020_split.csv')
        self.split_file = pd.read_csv(split_path, delimiter=',')
        self.annotations = pd.read_csv(metadata_path, delimiter=',')
        split_id = self.split_file['motion-id'][self.split_file['set'].str.lower() == split]
        self.annotations = self.annotations[self.annotations['motion-id'].isin(split_id)]

        with open('wav_corrections.txt', 'r') as f:
            corr = f.read().splitlines()
        for c in corr:
            c = c.split(' ')
            self.annotations['wav-file-name'].replace(c[0], c[1], inplace=True)
            assert c[1] in self.annotations['wav-file-name'].values
            assert c[0] not in self.annotations['wav-file-name'].values

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        text_path = os.path.join(self.path, 'trs.txt', self.annotations['clean-transcript-file-name'].iloc[idx])
        audio_path = os.path.join(self.path, 'wav', self.annotations['wav-file-name'].iloc[idx])

        with open(text_path, 'r') as f:
            text = f.read()
            motion = self.annotations['motion'].iloc[idx]
        encoded_dict = self.tokenizer(motion, text, truncation=True, max_length=512, verbose=False)
        ids = encoded_dict['input_ids']
        seg = encoded_dict['token_type_ids']
        att = encoded_dict['attention_mask']

        audio = []#torchaudio.load(audio_path)

        if self.text_transform:
            ids = self.text_transform(ids)
            seg = self.text_transform(seg)
            att = self.text_transform(att)
        if self.audio_transform:
            audio = self.audio_transform(audio)
        
        label = self.annotations['speech-to-motion-polarity'].iloc[idx]
        if label == 'pro':
            label = 1.0
        else:
            label = 0.0
        return [ids, seg, att], audio, label
