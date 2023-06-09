import os
import torch
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_some_examples(real, fake, epoch, folder):
    save_path = os.path.join(folder, f'Epoch_{epoch}')
    
    if not os.path.exists(save_path):
        os.system(f"mkdir {save_path}")
    
    save_image(fake, save_path + f"/y_gen_{epoch}.png")
    save_image(real, save_path + f"/input_{epoch}.png")

def save_checkpoint(model, optimizer, filename = "my_checkpoint.pt"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location = device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr