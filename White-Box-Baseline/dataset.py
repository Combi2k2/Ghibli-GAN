import os
import torch
import config
import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5], #[104.49958813/255, 111.69546534/255, 97.71923648/255],
        [0.5, 0.5, 0.5]#[103.88423806/255, 110.8071049/255,  96.9613091/255]
    ),
])

class GhibliDataset(Dataset):
    def __init__(self, input_dir, target_dir, size):
        self.input_dir = input_dir
        self.target_dir = target_dir
        
        self.input_files = dict()
        self.target_files = dict()
        
        for label in config.LABELS:
            self.input_files[label] = os.listdir(os.path.join(input_dir, label))[:size//len(config.LABELS)]
            self.target_files[label] = os.listdir(os.path.join(target_dir, label))[:size//len(config.LABELS)]
    
    def __len__(self):
        return sum([len(self.input_files[label]) for label in config.LABELS])

    def __getitem__(self, index):
        label = config.LABELS[index % 4]
        
        input_file = self.input_files[label][index // len(config.LABELS)]
        target_file = self.target_files[label][index // len(config.LABELS)]
        
        input_path = os.path.join(os.path.join(self.input_dir, label), input_file)
        target_path = os.path.join(os.path.join(self.target_dir, label), target_file)
        
        inputs = transform(Image.open(input_path))
        target = transform(Image.open(target_path))
        
        return inputs, target

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

import numpy as np

transforms_ = [
    transforms.Resize(int(256*1.12), Image.BICUBIC),
    transforms.RandomCrop((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

class AnimeFaceDataset(Dataset):
    def __init__(self, transforms_=None, unaligned=False, mode='train'):
        self.transform = transform
        self.unaligned = unaligned
        self.mode = mode
        if self.mode == 'train':
            self.files_A = sorted(glob.glob('./../CycleGAN/dataset/cartoon/*/*.*')[:1000])
            self.files_B = sorted(glob.glob('./../CycleGAN/dataset/photo/*.*')[:1000])
        elif self.mode == 'test':
            self.files_A = sorted(glob.glob('./../CycleGAN/dataset/cartoon/*/*.*')[1000:])
            self.files_B = sorted(glob.glob('./../CycleGAN/dataset/photo/*.*')[1000:])
        # if self.mode == 'train':
        #     self.files_A = sorted(glob.glob(os.path.join(root+'/dataset/cartoon/*')+'/*.*')[:250])
        #     self.files_B = sorted(glob.glob(os.path.join(root+'/dataset/photo')+'/*.*')[:250])
        # elif self.mode == 'test':
        #     self.files_A = sorted(glob.glob(os.path.join(root+'/dataset/cartoon/*')+'/*.*')[250:])
        #     self.files_B = sorted(glob.glob(os.path.join(root+'/dataset/photo')+'/*.*')[250:301])

    def  __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        if self.unaligned:
            image_B = Image.open(self.files_B[np.random.randint(0, len(self.files_B)-1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])
        if image_A.mode != 'RGB':
            image_A = to_rgb(image_A)
        if image_B.mode != 'RGB':
            image_B = to_rgb(image_B)
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        return item_B, item_A
        # return {'A':item_A, 'B':item_B}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

if __name__ == '__main__':
    ds = GhibliDataset('data/Cartoon', 'data/Real-Images', size = 400)
    
    import torchvision.transforms as T

    input, target = ds[0]
    print(torch.min(input), torch.max(input))
    print(torch.min(target), torch.max(target))
    transform = T.ToPILImage()
    # transform((input + 1) * 127.5).show()
    # transform((target + 1) * 127.5).show()