from torchvision.utils import save_image
import numpy as np
import torch
import cv2

import os

import config

def save_some_examples(real, fake, epoch, folder):
    save_path = os.path.join(folder, f'Epoch_{epoch}')
    
    if not os.path.exists(save_path):
        os.system(f"mkdir {save_path}")
    
    save_image(fake, save_path + "/fake.png")
    save_image(real, save_path + "/real.png")

def save_checkpoint(model_object, filename = "my_checkpoint.pt"):
    if config.CHECKPOINT_DIR not in filename:
        filename = os.path.join(config.CHECKPOINT_DIR, filename)
    
    print("=> Saving checkpoint")
    torch.save(model_object.state_dict(), filename)

def load_checkpoint(model_object, filename, lr = None):
    if config.CHECKPOINT_DIR not in filename:
        filename = os.path.join(config.CHECKPOINT_DIR, filename)
    
    if lr:  print("=> Loading checkpoint for optimizer")
    else:   print("=> Loading checkpoint for model")
    
    checkpoint = torch.load(filename, map_location = config.DEVICE)
    model_object.load_state_dict(checkpoint)
    
    if lr is None:
        return
    
    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in model_object.param_groups:
        param_group["lr"] = lr

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cam(x, size = 256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

def tensor2img(img_tensor):
    return RGB2BGR(tensor2numpy(denorm(img_tensor)))

def tensor2img_with_heatmap(img_tensor, heat_map):
    image_np = RGB2BGR(tensor2numpy(denorm(img_tensor)))
    heat_map = cam(tensor2numpy(heat_map), img_tensor.shape[2])
    
    return image_np * 0.8 + heat_map * 0.2
    