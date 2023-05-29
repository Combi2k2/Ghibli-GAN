import torchvision.transforms as transforms
import torch
import cv2

from model import GeneratorResNet
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
    generator = GeneratorResNet((channels, img_height, img_width), 9).to(device)
    generator.load_state_dict(torch.load('checkpoints/Generator_B2A.pt').state_dict())
    generator.eval()

    img = Image.open('dataset/photo/00094.jpg')
    img = transform(img)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = generator(img)[0]
        pred = pred * 0.5 + 0.5   # remove normalization #
    
    pred = (pred.permute(1, 2, 0).cpu().numpy() + 1) * 127.5

    cv2.imwrite("test.jpg", pred)


