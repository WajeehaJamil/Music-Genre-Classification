from pickle import load as pickle_load
from pathlib import Path
from matplotlib import transforms
import torch
import numpy as np

class GTZANDataset:
    def __init__(self, df, transform=None):
        self.paths = np.array(df['feature_path'])
        self.transforms = transform

    def load_file(self,file_path):
        with file_path.open('rb') as f:
            dict = pickle_load(f)
        return dict['features'], dict['class']

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
       path = self.paths[idx]
       features, target = self.load_file(Path(path))
       if self.transforms is not None:
           features = self.transforms(features)
           
       target = np.array(target.split(',')).astype('float')
       return features, torch.tensor(target)