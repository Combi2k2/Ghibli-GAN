import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import config

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    ),
])

class CartoonDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self._len = 0
        
        self.input_files = dict()
        self.target_files = dict()
        
        for label in config.LABELS:
            self.input_files[label] = os.listdir(os.path.join(input_dir, label))
            self.target_files[label] = os.listdir(os.path.join(target_dir, label))
            
            self._len += max(
                len(self.input_files[label]),
                len(self.target_files[label])
            )
    
    def __len__(self):
        return self._len

    def __getitem__(self, index):
        label = config.LABELS[index % len(config.LABELS)]
        index = index // len(config.LABELS)
        
        input_file = self._get_sample(self.input_files[label], index)
        target_file = self._get_sample(self.target_files[label], index)
        
        input_path = os.path.join(os.path.join(self.input_dir, label), input_file)
        target_path = os.path.join(os.path.join(self.target_dir, label), target_file)
        
        inputs = transform(Image.open(input_path))
        target = transform(Image.open(target_path))
        
        return inputs, target

    def _get_sample(self, files, index):
        return files[index % len(files)]

if __name__ == '__main__':
    ds = CartoonDataset('./../data/Cartoon', './../data/Real-Images')
    
    import torchvision.transforms as T

    input, target = ds[457]
    print(torch.min(input), torch.max(input))
    print(torch.min(target), torch.max(target))
    transform = T.ToPILImage()
    
    transform((input + 1) * 127.5).show()
    transform((target + 1) * 127.5).show()