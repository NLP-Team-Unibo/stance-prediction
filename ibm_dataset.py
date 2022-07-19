from genericpath import sameopenfile
import os
import ast
import pickle
import librosa
import pandas as pd

import torch
import torchaudio
from torch.utils.data import Dataset
from utils.data import truncate_encoded_text

class IBMDebater(Dataset):
    def __init__(
            self, 
            path, 
            split, 
            tokenizer=None, 
            chunk_length=15, 
            text_transform=None, 
            load_audio=True, 
            load_text=True,
            sample_cut_type='first',
            load_motion=False,
            load_audio_emb=False,
            audio_emb_base_path='out'
        ):
        """
            Custom Pytorch dataset class for reading the IBM Debater 'Debate Speech Analysis' dataset, which can be freely downloaded 
            and accessed at https://research.ibm.com/haifa/dept/vst/debating_data.shtml. This class allows to load only the text
            transcript, only the audio or both; it also allows text tokenization and transformation using torchtext.transforms.
            Notice that the audio files are always cut after chunk_lenght seconds, independently from their actual duration.
    
            Parameters
            ----------
            path: str
                Path to the local folder containing the IBM debater dataset.
            split: str
                Which dataset split to load, it should be either 'train' or 'test'.
            tokenizer:  transformers.DistilBertTokenizer
                The tokenizer, if any, to use in order to pre-process the input text before passing it to the model. Default to None.
            chunk_lenght: int
                The lenght in seconds at which to cut each audio file when loading them. Deafult to 15.
            text_transform: torchtext.transforms
                The transformations, if any, to apply to the text before passing it to the model. Default to None.
            load_audio: bool
                Whether to load the audio files or not. Default to True.
            load_text: bool
                Whether to load the text sentences or not. Default to True. 
            
        """

        self.path = path
        self.tokenizer = tokenizer
        self.text_transform = text_transform
        self.load_audio = load_audio
        self.load_text = load_text
        self.chunk_length = chunk_length
        self.sample_cut_type = sample_cut_type
        self.text_gen = load_motion
        self.load_audio_emb = load_audio_emb
        self.audio_emb_base_path = audio_emb_base_path

        # Loading the csv files containing the dataset splits and the additional information for eaach element
        metadata_path = os.path.join(path, 'RecordedDebatingDataset_Release5_metadata.csv')
        split_path = os.path.join(path, 'OutOfTheEchoChamber_acl2020_split.csv')
        self.split_file = pd.read_csv(split_path, delimiter=',')
        self.annotations = pd.read_csv(metadata_path, delimiter=',')

        # Only select the elements that are in the required split
        split_id = self.split_file['motion-id'][self.split_file['set'].str.lower() == split]
        self.annotations = self.annotations[self.annotations['motion-id'].isin(split_id)]

        # Some audio files do not have a corresponding element in the csv files. Since we found that these discrepancies
        # were most probably due to spelling mistakes, we choose to re-name the annotations on the fly according to the
        # mappings shown in the file wav_corrections.txt
        if split == 'train' or split == 'validation':
            with open('wav_corrections.txt', 'r') as f:
                data = f.read()
            corrections = ast.literal_eval(data)
            for c in corrections[split]:
                self.annotations['wav-file-name'].replace(c[0], c[1], inplace=True)
                assert c[1] in self.annotations['wav-file-name'].values
                assert c[0] not in self.annotations['wav-file-name'].values

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        text_path = os.path.join(self.path, 'trs.txt', self.annotations['clean-transcript-file-name'].iloc[idx])
        audio_path = os.path.join(self.path, 'wav', self.annotations['wav-file-name'].iloc[idx])
        audio_emb_path = os.path.join(self.audio_emb_base_path, f'{self.chunk_length}-{self.sample_cut_type}', self.annotations['wav-file-name'].iloc[idx].replace('wav', 'pkl'))
        output = []
        # Load the text corresponding to idx and apply tokenization or any other transformation that was specified 
        # at initialization time
        is_multimodal = 'multimodal' in self.tokenizer.__class__.__name__.lower()
        if self.load_text:
            with open(text_path, 'r') as f:
                text = f.read()    
            if self.tokenizer and not is_multimodal:
                text = self.tokenizer(text, truncation=False)
                text = truncate_encoded_text(encoded_text=text, mode=self.sample_cut_type)
                if self.text_transform:
                    for k in text.keys():
                        text[k] = self.text_transform(text[k])
            output.append(text)

        if self.text_gen:
            motion = self.annotations['motion'].iloc[idx]
            if self.tokenizer:
                if is_multimodal:
                    motion = self.tokenizer.encode_label(motion)
                else:
                    motion = self.tokenizer(motion, truncation=True, return_tensors='pt')
                for k in motion.keys():
                    motion[k] = motion[k].squeeze()
            output.append(motion)
        
        # Load wav file corresponding to idx and resample to 16000, which is the sample rate that Wav2Vec2
        # expectes as input
        if self.load_audio:
            audio_len = librosa.get_duration(filename=audio_path)
            if self.sample_cut_type == 'first':
                wave, sr = librosa.load(audio_path, duration=self.chunk_length)
                wave = torch.tensor(wave)
            elif self.sample_cut_type == 'last':
                wave, sr = librosa.load(audio_path, offset=audio_len - self.chunk_length)
                wave = torch.tensor(wave)
            if self.sample_cut_type == 'both':
                first_wave, sr = librosa.load(audio_path, duration=self.chunk_length / 2)
                second_wave, _ = librosa.load(audio_path, offset=audio_len - self.chunk_length / 2)
                first_wave = torch.tensor(first_wave)
                second_wave = torch.tensor(second_wave)
                wave = torch.cat([first_wave, second_wave])

            wave = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wave)
            
            target = self.chunk_length * 16000

            if len(wave) < target:
                wave = torch.nn.functional.pad(wave, (0, target - len(wave)))
            else:
                wave = wave[:target]
            output.append(wave)
        elif self.load_audio_emb:
            with open(audio_emb_path, 'rb') as audio_features_file:
                audio_feat = pickle.load(audio_features_file)
            output.append(audio_feat)
        
            num_audio_features = audio_feat.shape[1]
            output[0] = self.tokenizer.encode_mult(num_audio_features, output[0], cut_mode=self.sample_cut_type)


        # Convert label to numeric format
        label = self.annotations['speech-to-motion-polarity'].iloc[idx]
        label = 1.0 if label == 'pro' else 0.0
        output.append(label)

        return output