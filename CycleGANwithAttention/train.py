
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from trainer import Trainer
from dataset import CartoonDataset

import logging
import config

# Set up dataset
print("Initializing dataset")

dataset = CartoonDataset('./../data/Cartoon', './../data/Real-Images')
train_ds, test_ds = train_test_split(dataset, test_size = 0.1)

train_loader = DataLoader(train_ds, batch_size = config.BATCH_SIZE, shuffle = True, num_workers = config.NUM_WORKERS)
test_loader = DataLoader(test_ds, batch_size = 1, num_workers = config.NUM_WORKERS)

# Set up the models
print("Initializing model")

trainer = Trainer()

# Train Loop
print("Training")
test_iter = iter(test_loader)

for i in range(config.NUM_EPOCHS):
    for batch_idx, sample in enumerate(train_loader):
        loss_dict = trainer.run_train(sample)
        
        if (batch_idx % 1 == 0):
            print(f"Epoch [{i}/{config.NUM_EPOCHS}] Batch {batch_idx} / {len(train_loader)}:")
            print(f"\t[D_loss: {loss_dict['D_loss']:.4f}] ", end = "")
            print(f" [G_loss: {loss_dict['G_loss']:.4f} - (", end = "")
            print(f"adv : {loss_dict['G_adv_loss']:.4f}, ", end = "")
            print(f"cycle : {loss_dict['G_recon_loss']:.4f}, ", end = "")
            print(f"identity : {loss_dict['G_recon_loss']:.4f}, ", end = "")
            print(f"cam : {loss_dict['G_cam_loss']:.4f})")
            
    try:
        test_images = next(test_iter)
    except:
        test_iter = iter(test_loader)
        test_images = next(test_iter)
    
    trainer.eval()
    trainer.run_test(f'evaluate/Epoch{i}')
    trainer.train()
    
    if (config.SAVE_CHECKPOINT):
        trainer.save()