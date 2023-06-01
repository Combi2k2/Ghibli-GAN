
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime

from trainer import Trainer
from dataset import CartoonDataset
from utils import count_parameters

import torch
import logging
import config
import os

# Set up dataset
print("Initializing dataset")

dataset = CartoonDataset('./../CycleGAN/dataset/photo', './../CycleGAN/dataset/cartoon')
train_ds, test_ds = train_test_split(dataset, test_size = 0.05)

train_loader = DataLoader(train_ds, batch_size = config.BATCH_SIZE, shuffle = True, num_workers = config.NUM_WORKERS)
test_loader = DataLoader(test_ds, batch_size = 4, num_workers = config.NUM_WORKERS)

# Set up the models
print("Initializing model")
trainer = Trainer()

def add_sn(m):
    if isinstance(m, torch.nn.Conv2d):
        return torch.nn.utils.spectral_norm(m)
    else:
        return m

trainer.discA.apply(add_sn)
trainer.discB.apply(add_sn)

if (config.LOAD_CHECKPOINT):
    trainer.load(config.CHECKPOINT_DIR)

print(">>  Number of generator's trainable params:", count_parameters(trainer.genA2B))
print(">>  Number of discriminator's trainable params:", count_parameters(trainer.discA))

# Train Loop
print("Training")
test_iter = iter(test_loader)
loss_history = []

if (config.LOAD_CHECKPOINT):
    loss_history = torch.load(os.path.join(config.CHECKPOINT_DIR, 'train_loss.pt'), map_location = config.DEVICE)

logging.basicConfig(
    filename = os.path.join(config.CHECKPOINT_DIR, 'train_history.log'),
    level = logging.INFO,
)
logging.captureWarnings(True)

for i in range(1, config.NUM_EPOCHS + 1):
    for batch_idx, sample in enumerate(train_loader):
        loss_dict = trainer.run_train(sample)
        loss_history.append(loss_dict)

        step = len(loss_history)

        if (step % 100 == 0 or batch_idx % 50 == 0):
            loss_string = f" [D_loss: {loss_dict['D_loss']:.4f}]"
            loss_string += f" [G_loss: {loss_dict['G_loss']['total_loss']:.4f} - ("
            loss_string += f"adv: {loss_dict['G_loss']['adv']:.4f}, "
            loss_string += f"cam: {loss_dict['G_loss']['cam']:.4f}, "
            loss_string += f"cycle: {loss_dict['G_loss']['cycle']:.4f}, "
            loss_string += f"identity: {loss_dict['G_loss']['identity']:.4f})"

            if (step % 100 == 0):
                logging.info(f"Step [{len(loss_history)}]: " + loss_string)
            
            if (batch_idx % 50 == 0):
                print(f'Epochs [{i} / {config.NUM_EPOCHS}] Batch [{batch_idx} / {len(train_loader)}]:', loss_string)
            
    try:
        test_images = next(test_iter)
    except:
        test_iter = iter(test_loader)
        test_images = next(test_iter)
    
    trainer.eval()
    trainer.run_test(test_images, f'evaluate/Epoch{i}')
    trainer.train()
    
    if (config.SAVE_CHECKPOINT):
        trainer.save()
        torch.save(loss_history, os.path.join(config.CHECKPOINT_DIR, 'train_loss.pt'))