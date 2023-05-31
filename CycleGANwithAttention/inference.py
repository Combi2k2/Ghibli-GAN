import torchvision.transforms as transforms
import torch
import cv2

from utils import load_checkpoint
from trainer import Trainer
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_height, img_width, channels = (256, 256, 3)

transform = transforms.Compose([
    transforms.Resize(int(img_height*1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == '__main__':
    img = Image.open('hoam.png')
    img = transform(img)
    img = img.unsqueeze(0).to(device)

    trainer = Trainer()

    load_checkpoint(trainer.genA2B, 'checkpoints/GeneratorA2B.pt')
    load_checkpoint(trainer.genB2A, 'checkpoints/GeneratorB2A.pt')

    trainer.eval()
    trainer.run_test(img, 'celeb', filename = 'hoaminzy.png', infer_mode = 'A2B')


