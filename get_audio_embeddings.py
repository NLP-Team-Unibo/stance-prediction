import os
import pickle
from sklearn.utils import shuffle
import tqdm
import argparse
from models.audio_model import AudioModel
from ibm_dataset import IBMDebater
from torch.utils.data import DataLoader
from utils.batch_generators import batch_generator_wav2vec

SPLITS = ['train', 'test', 'validation']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=False, default='full/',
                        help='input directory')
    parser.add_argument('--output_dir', type=str, required=False, default='out/',
                        help='output directory')
    parser.add_argument('--device', type=str, required=False, default='cuda',
                        help='device (cpu or gpu) to load data and model on')
    parser.add_argument('--chunk_length', type=int, required=False, default=10,
                        help='audio chunk length')
    parser.add_argument('--cut_type', type=str, required=False, default='last',
                        help='whether to take the first, the last or a mix of both chunks of the audio file')
    args = parser.parse_args()

    out_dir = os.path.join(args.output_dir, f'{args.chunk_length}-{args.cut_type}')
    model = AudioModel(n_trainable_layers=0, return_sequences=True)
    model.to(args.device)
    model.eval()
    if os.path.exists(out_dir):
        print('Using previously computed embeddings!')
    else:
        os.makedirs(out_dir)
        for split in SPLITS:
            dataset = IBMDebater(path=args.input_dir, split=split, load_text=False, load_audio=True, chunk_length=args.chunk_length, sample_cut_type=args.cut_type)
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=batch_generator_wav2vec)
            for audio, filename in tqdm.tqdm(zip(data_loader, dataset.annotations['wav-file-name']), desc=f'Processing {split} split..'):
                features = model(audio[0].to(args.device))
                filename = os.path.join(out_dir, str(filename).replace('wav', 'pkl'))
                pickle.dump(features.to('cpu'), open(filename, 'wb'))
                #with open(filename, 'rb') as f:
                #    x = pickle.load(f)
                #print(features.shape, x.shape)
                #input()

    
