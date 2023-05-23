import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5], #[104.49958813/255, 111.69546534/255, 97.71923648/255],
        [0.5, 0.5, 0.5]#[103.88423806/255, 110.8071049/255,  96.9613091/255]
    ),
])

LABELS = ['City', 'Human', 'Scenery', 'Room']

class GhibliDataset(Dataset):
    def __init__(self, input_dir, target_dir, size):
        self.input_dir = input_dir
        self.target_dir = target_dir
        
        self.input_files = dict()
        self.target_files = dict()
        
        for label in LABELS:
            self.input_files[label] = os.listdir(os.path.join(input_dir, label))[:size//4]
            self.target_files[label] = os.listdir(os.path.join(target_dir, label))[:size//4]
    
    def __len__(self):
        return sum([len(self.input_files[label]) for label in LABELS])

    def __getitem__(self, index):
        label = LABELS[index % 4]
        
        input_file = self.input_files[label][index // 4]
        target_file = self.target_files[label][index // 4]
        
        input_path = os.path.join(os.path.join(self.input_dir, label), input_file)
        target_path = os.path.join(os.path.join(self.target_dir, label), target_file)
        
        inputs = transforms(Image.open(input_path))
        target = transforms(Image.open(target_path))
        
        return inputs, target

if __name__ == '__main__':
    ds = GhibliDataset('data/Cartoon', 'data/Real-Images')
    
    import torchvision.transforms as T

    input, target = ds[0]
    print(torch.min(input), torch.max(input))
    print(torch.min(target), torch.max(target))
    transform = T.ToPILImage()
    transform(input).show()
    transform(target).show()