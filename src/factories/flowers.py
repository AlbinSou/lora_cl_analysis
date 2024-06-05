#!/usr/bin/env python3
#!/usr/bin/env python3
import torch
import scipy.io
import PIL
from torch.utils.data import Dataset
import os
import numpy as np

def convert_pil(filename):
    image = PIL.Image.open(filename)
    return image

class FlowersDataset(Dataset):
    """ Flowers 102 Dataset torchvision """

    SPLITS = {'train': 'trnid', 'valid': 'valid', 'test': 'tstid'}
    
    def __init__(self, root_dir, split='train'):
        super().__init__()
        self.root = os.path.join(root_dir, '102flowers')
        assert os.path.isdir(self.root)
        self.split_index = self._load_splits(os.path.join(self.root, 'setid.mat'), split)
        self.labels = self._load_labels(os.path.join(self.root, 
                                        'imagelabels.mat'))
        self.targets = self.labels[self.split_index - 1].astype(np.int64) - 1

    def __getitem__(self, idx):
        index = self.split_index[idx]
        label = self.targets[idx]
        image = convert_pil(os.path.join(self.root, 
                            'jpg', f'image_{index:05}.jpg'))
        return image, label

    def _load_labels(self, filename):
        labels = scipy.io.loadmat(filename)['labels'][0]
        return labels

    def _load_splits(self, filename, mode):
        mat = scipy.io.loadmat(filename)
        split_index = mat[self.SPLITS[mode]][0]
        return split_index

    def __len__(self):
        return len(self.split_index)
