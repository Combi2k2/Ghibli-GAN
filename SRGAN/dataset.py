import numpy as np
import config
import os

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from os.path import dirname, abspath


FILE_DIR = dirname(abspath(__file__))
REPO_DIR = dirname(FILE_DIR)

class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        file_path = os.path.join(self.root_dir, file_name)
        
        image = np.array(Image.open(file_path))
        image = config.both_transforms(image = image)["image"]
        high_res = config.highres_transform(image = image)["image"]
        low_res = config.lowres_transform(image = image)["image"]
        return low_res, high_res

if __name__ == "__main__":
    dataset = MyImageFolder(root_dir="./../data/Cartoon/Human")
    
    print(len(dataset))
    
    low_res, high_res = dataset[0]
    
    print(low_res.shape)
    print(high_res.shape)