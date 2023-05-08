import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    ),
])

# Dataset source: https://www.kaggle.com/datasets/zhouxm/anime-sketch-colorization

class ColorizationDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        
        self.list_files = os.listdir(input_dir)
    
    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        
        input_path = os.path.join(self.input_dir, img_file)
        target_path = os.path.join(self.target_dir, img_file)
        
        input = transforms(Image.open(input_path))
        target = transforms(Image.open(target_path))
        
        return input, target

if __name__ == '__main__':
    ds = ColorizationDataset('data/testB', 'data/testA')
    
    input, target = ds[0]
    
    print(input.shape)
    print(target.shape)