import torchvision.transforms as transforms
import torch
import cv2
import config

from torchvision.utils import save_image
from filters import guided_filter, color_shift, simple_superpixel
from utils import load_checkpoint
from model import Unet_Generator
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_height, img_width, channels = (256, 256, 3)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == '__main__':
    img = Image.open('./../CycleGAN/dataset/photo/Human/0000.jpg')
    img = transform(img)
    img = img.unsqueeze(0).to(device)

    CHECKPOINT_PATH = 'checkpoints/checkpoints_selfie2anime/Generator.pt'
    checkpoint = torch.load(CHECKPOINT_PATH, map_location = device)

    model = Unet_Generator(channels = config.CHANNELS_IMG, features = config.FEATURES_GEN, num_blocks = 4).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    fake = model(img)

    blur_fake = guided_filter(fake, fake, r = 5, eps = 0.2)
    gray_fake, _ = color_shift(fake, img)
    gray_fake = gray_fake.expand(1, 3, 256, 256)

    print(blur_fake.shape)
    print(gray_fake.shape)

    real_superpixel = torch.from_numpy(simple_superpixel(img.permute(0, 2, 3, 1).cpu().numpy())).permute(0, 3, 1, 2).to(device)

    print(real_superpixel.shape)
    
    images = torch.cat([img, fake, real_superpixel, blur_fake, gray_fake], dim = 0) * 0.5 + 0.5
    
    save_image(images, "inference/image.png")




