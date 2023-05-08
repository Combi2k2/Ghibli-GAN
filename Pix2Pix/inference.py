import os
import sys
import torch
from PIL import Image
from torchvision.utils import save_image

from model import Generator
from dataset import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading model checkpoint

CHECKPOINT_DIR = 'checkpoints/'
CHECKPOINT_GEN = os.path.join(CHECKPOINT_DIR, 'Generator.pt')

print("=> Loading checkpoint")

checkpoint = torch.load(CHECKPOINT_GEN, map_location = device)

gen = Generator().to(device)
gen.load_state_dict(checkpoint["state_dict"])
gen.eval()

# parsing command arguments

assert(len(sys.argv) > 1, "Pass an image as input")

img_file = sys.argv[-1]
img = transforms(Image.open(img_file))
img = torch.unsqueeze(img, 0).to(device)

print(img.shape)

# generating output

with torch.no_grad():
    y_fake = gen(img)
    y_fake = y_fake * 0.5 + 0.5  # remove normalization #
    
    save_image(y_fake, "picture-color.jpg")

