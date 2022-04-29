from utils import * 
from preprocessing.feature_extract import extract_mel_band_energies
import matplotlib.pyplot as plt
import pandas as pd
from pickle import load as pickle_load
from pathlib import Path
import librosa
from dataset.GTZANDataset import GTZANDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from model.net import RnnModel
from preprocessing.serialize_data import serialize_split_data

def main():
    file_path = 'data/original_audio/blues/blues.00000.au'
    audio, sr = get_audio_file_data(file_path)
    print(sr)
    print(np.shape(audio))
    f = extract_mel_band_energies(audio)
    print(np.shape(f))
    plt.imshow(f)
    plt.show()
    
def test():
    test_csv = pd.read_csv('data/test_metadata.csv')
    train_csv = pd.read_csv('data/train_metadata.csv')
    file_path = Path(train_csv['feature_path'][100])
    
    with file_path.open('rb') as f:
        dict = pickle_load(f)
        feature, label = dict['features'], dict['class']
    plt.imshow(feature)
    plt.show()
    
    print(feature.shape)
    
def tt():
    train_df = load_csv(cfg['paths']['train_csv_path'])
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((40,cfg['feature_extract']['frames'])),
            transforms.RandomHorizontalFlip(p=0.9),
        ])
    
    net = RnnModel()
    dataset = GTZANDataset(train_df, transform)
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for x, y in dataloader:
        x = x.reshape(1,1000,40)
        print(x.shape)
        print(y.shape)
        y_out = net(x)
        print(y_out)
        break
    
    
if __name__ == "__main__":
    cfg = read_yaml()
    
    #df = create_csv(cfg['class_names'], cfg['paths']['split_audio_path'])
    #print(df.head())
    #test()
    serialize_split_data(cfg)