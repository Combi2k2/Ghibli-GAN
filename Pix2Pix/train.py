import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Discriminator, Generator
from dataset import ColorizationDataset
from utils import save_some_examples, save_checkpoint, load_checkpoint, device

# Hyperparameters etc.
LEARNING_RATE = 5e-5  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS_IMG = 3
NUM_EPOCHS = 100
# FEATURES_DISC = 64
# FEATURES_GEN = 64

SAVE_MODEL = True
LOAD_CHECKPOINT = True
CHECKPOINT_DIR = 'checkpoints/'

CHECKPOINT_GEN = os.path.join(CHECKPOINT_DIR, 'Generator.pt')
CHECKPOINT_DISC = os.path.join(CHECKPOINT_DIR, 'Discriminator.pt')

# Initialize the models

gen = Generator().to(device)
disc = Discriminator().to(device)

opt_gen = optim.Adam(gen.parameters(), lr = LEARNING_RATE, betas = (0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr = LEARNING_RATE, betas = (0.5, 0.999))

g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

BCE_LOSS = nn.BCEWithLogitsLoss()
L1_LOSS = nn.L1Loss()

if (LOAD_CHECKPOINT):
    load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE)
    load_checkpoint(CHECKPOINT_DISC, disc, opt_disc, LEARNING_RATE)

# Initialize the dataset

train_ds = ColorizationDataset('data/trainB', 'data/trainA')
valid_ds = ColorizationDataset('data/testB', 'data/testA')

train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True, num_workers = 8)
valid_loader = DataLoader(valid_ds, batch_size = BATCH_SIZE, num_workers = 8)

# initialize_weights(gen)
# initialize_weights(disc)

for epoch in range(NUM_EPOCHS):
    loop = tqdm(train_loader, leave = True)
    
    for idx, (x, y) in enumerate(loop):
        x = x.to(device)
        y = y.to(device)
        
        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())
            
            D_real_loss = BCE_LOSS(D_real, torch.ones_like(D_real))
            D_fake_loss = BCE_LOSS(D_fake, torch.zeros_like(D_fake))
            
            D_loss = (D_real_loss + D_fake_loss) / 2
        
        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        
        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_loss = BCE_LOSS(D_fake, torch.ones_like(D_fake)) + 100 * L1_LOSS(y_fake, y)

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
    if (SAVE_MODEL and (epoch + 1) % 10 == 0):
        if (not os.path.exists(CHECKPOINT_GEN)):
            os.system('mkdir -p ' + CHECKPOINT_DIR)
        
        save_checkpoint(gen, opt_gen, CHECKPOINT_GEN)
        save_checkpoint(disc, opt_disc, CHECKPOINT_DISC)
    
    save_some_examples(gen, valid_loader, epoch, 'eval')