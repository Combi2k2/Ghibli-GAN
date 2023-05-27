from PIL import Image
import torchvision.transforms as transforms
import os
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

# data (path)
dataset_name = 'gan-getting-started'
root = '../input/'+dataset_name
img_height, img_width, channels = (256, 256, 3)
transforms_ = [
    transforms.Resize(int(img_height*1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.mode = mode
        if self.mode == 'train':
            self.files_A = sorted(glob.glob('dataset/cartoon/*/*.*')[:1000])
            self.files_B = sorted(glob.glob('dataset/photo/*.*')[:1000])
        elif self.mode == 'test':
            self.files_A = sorted(glob.glob('dataset/cartoon/*/*.*')[1000:])
            self.files_B = sorted(glob.glob('dataset/photo/*.*')[1000:])
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
        return {'A':item_A, 'B':item_B}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
n_cpu = 3

dataloader = DataLoader(
    ImageDataset(root, transforms_=transforms_, unaligned=True),
    batch_size=8, # 1
    shuffle=True,
    num_workers=n_cpu # 3
)

val_dataloader = DataLoader(
    ImageDataset(root, transforms_=transforms_, unaligned=True, mode='test'),
    batch_size=8,
    shuffle=True,
    num_workers=n_cpu
)