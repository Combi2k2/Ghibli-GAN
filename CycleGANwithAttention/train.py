
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

dataset = CartoonDataset('./../data/Cartoon', './../data/Real-Images')
train_ds, test_ds = train_test_split(dataset, test_size = 0.1)

train_loader = DataLoader(train_ds, batch_size = config.BATCH_SIZE, shuffle = True, num_workers = config.NUM_WORKERS)
test_loader = DataLoader(test_ds, batch_size = 4, num_workers = config.NUM_WORKERS)

# Set up the models
print("Initializing model")
trainer = Trainer()

print(">>  Number of generator's trainable params:", count_parameters(trainer.genA2B))
print(">>  Number of discriminator's trainable params:", count_parameters(trainer.discA))

# Train Loop
print("Training")
test_iter = iter(test_loader)
loss_history = []

logging.basicConfig(
    filename = os.path.join(config.CHECKPOINT_DIR, 'train_history.log'),
    level = logging.INFO,
)
logging.captureWarnings(True)

for i in range(config.NUM_EPOCHS):
    logging.info(f'Epoch [{i}/{config.NUM_EPOCHS}]: Start at {datetime.now()}')
    
    for batch_idx, sample in enumerate(train_loader):
        loss_dict = trainer.run_train(sample)
        loss_history.append(loss_dict)
        
        if (batch_idx % 1 == 0):
            loss_string = f"    Batch [{batch_idx + 1} / {len(train_loader)}]:"
            loss_string += f" [D_loss: {loss_dict['D_loss']:.4f}]"
            loss_string += f" [G_loss: {loss_dict['G_loss']['total_loss']:.4f} - ("
            loss_string += f"adv: {loss_dict['G_loss']['adv']:.4f}, "
            loss_string += f"cam: {loss_dict['G_loss']['cam']:.4f}, "
            loss_string += f"cycle: {loss_dict['G_loss']['cycle']:.4f}, "
            loss_string += f"identity: {loss_dict['G_loss']['identity']:.4f})"
            
            logging.info(loss_string)
            print(f'Epochs [{i}/{config.NUM_EPOCHS}]', loss_string)
            
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