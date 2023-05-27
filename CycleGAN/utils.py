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

def save_checkpoint(state_dict, filename = "my_checkpoint.pt"):
    print("=> Saving checkpoint")
    torch.save(state_dict, filename)

def load_checkpoint_model(checkpoint_file, model):
    print("=> Loading checkpoint model")
    checkpoint = torch.load(checkpoint_file, map_location = device)
    model.load_state_dict(checkpoint)

def load_checkpoint_optim(checkpoint_file, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location = device)
    optimizer.load_state_dict(checkpoint)

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)