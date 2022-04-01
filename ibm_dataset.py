from torch.utils.data import Dataset
import os
import pandas as pd
import torchaudio
import torch

class IBMDebater(Dataset):
    def __init__(self, path, split, tokenizer=None, audio_bundle=None, max_audio_len=25, text_transform=None, audio_transform=None, load_audio=True, load_text=True,):
        self.path = path
        self.tokenizer = tokenizer
        self.audio_bundle = audio_bundle
        self.text_transform = text_transform
        self.audio_transform = audio_transform
        self.load_audio = load_audio
        self.load_text = load_text
        self.max_audio_len = max_audio_len
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
        output = []
        if self.load_text:
            with open(text_path, 'r') as f:
                text = f.read()
                motion = self.annotations['motion'].iloc[idx]
            if self.tokenizer:
                text = self.tokenizer(motion, text, truncation=True, max_length=512)
            if self.text_transform:
                for k in text.keys():
                    text[k] = self.text_transform(text[k])
            output.append(text)

        if self.load_audio:
            wave, sr = torchaudio.load(audio_path)
            wave = torchaudio.functional.resample(wave, sr, self.audio_bundle.sample_rate)
            wave = wave[:, :self.max_audio_len*self.audio_bundle.sample_rate]
            wave = torch.mean(wave, dim=0)
            output.append(wave)
        
        label = self.annotations['speech-to-motion-polarity'].iloc[idx]
        if label == 'pro':
            label = 1.0
        else:
            label = 0.0
        output.append(label)
        return output
