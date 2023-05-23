import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import model
import loss

from filters import guided_filter, color_shift
from utils import save_some_examples, save_checkpoint, load_checkpoint, device
from dataset import GhibliDataset

print(device)

LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 16
IMAGE_SIZE = 256
CHANNELS_IMG = 3
NUM_EPOCHS = 100
NUM_WORKERS = 8

FEATURES_DISC = 32
FEATURES_GEN = 32

SAVE_MODEL = True
LOAD_CHECKPOINT = False
CHECKPOINT_DIR = 'checkpoints/'

if __name__ == '__main__':
    # Initialize the models
    gen = model.Unet_Generator(channels = CHANNELS_IMG, features = FEATURES_GEN, num_blocks = 4).to(device)
    disc_sf = model.Discriminator(channels = CHANNELS_IMG, features = FEATURES_DISC, patch = True).to(device)
    disc_tx = model.Discriminator(channels = 1, features = FEATURES_DISC, patch = True).to(device)
    
    # g_scaler = torch.cuda.amp.GradScaler()
    # d_scaler = torch.cuda.amp.GradScaler()

    opt_gen = optim.Adam(gen.parameters(), lr = LEARNING_RATE, betas = (0.5, 0.99))
    opt_disc_sf = optim.Adam(disc_sf.parameters(), lr = LEARNING_RATE, betas = (0.5, 0.99))
    opt_disc_tx = optim.Adam(disc_tx.parameters(), lr = LEARNING_RATE, betas = (0.5, 0.99))

    if not os.path.exists(CHECKPOINT_DIR):
        os.system(f'mkdir {CHECKPOINT_DIR}')

    if LOAD_CHECKPOINT == True:
        CHECKPOINT_GEN = os.path.join(CHECKPOINT_DIR, 'Generator.pt')
        CHECKPOINT_DISC_TX = os.path.join(CHECKPOINT_DIR, 'Discriminator_texture.pt')
        CHECKPOINT_DISC_SF = os.path.join(CHECKPOINT_DIR, 'Discriminator_surface.pt')
        
        load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE)
        load_checkpoint(CHECKPOINT_DISC_TX, disc_tx, opt_disc_tx, LEARNING_RATE)
        load_checkpoint(CHECKPOINT_DISC_SF, disc_sf, opt_disc_sf, LEARNING_RATE)

    print("Finish initializing model")

    # Initialize the dataset
    dataset = GhibliDataset('data/Real-Images', 'data/Cartoon', 1000)

    train_ds, valid_ds = train_test_split(dataset, test_size = 0.1, shuffle = True)

    train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS)

    val_iterator = iter(valid_loader)

    print("Finish initializing dataset")

    # Start training
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, cartoon) in enumerate(train_loader):
            print(batch_idx)
            input_photo = real.to(device)
            input_cartoon = cartoon.to(device)
            
            # Train Discriminator
            with torch.cuda.amp.autocast():
                output = gen(input_photo)
                output = guided_filter(input_photo, output, r = 1)
                
                blur_fake = guided_filter(output, output, r = 5, eps = 0.2)
                blur_cartoon = guided_filter(input_cartoon, input_cartoon, r = 5, eps = 0.2)
                
                gray_fake, gray_cartoon = color_shift(output, input_cartoon)
                
                # print(gray_fake.shape, gray_cartoon.shape)
                # print(blur_fake.shape, blur_cartoon.shape)
                
                d_loss_gray, _ = loss.lsgan_loss(disc_tx, gray_cartoon, gray_fake)
                d_loss_blur, _ = loss.lsgan_loss(disc_sf, blur_cartoon, blur_fake)
                
                opt_disc_tx.zero_grad();    d_loss_gray.backward(retain_graph=True)
                opt_disc_tx.step()
                
                opt_disc_sf.zero_grad();    d_loss_blur.backward(retain_graph=True)
                opt_disc_sf.step()
                
                # disc_tx.zero_grad()
                # d_scaler.scale(d_loss_gray).backward()
                # d_scaler.step(opt_disc_tx)
                # d_scaler.update()
                
                # disc_sf.zero_grad()
                # d_scaler.scale(d_loss_blur).backward()
                # d_scaler.step(opt_disc_sf)
                # d_scaler.update()
            
            disc_sf.eval()
            disc_tx.eval()
            
            with torch.cuda.amp.autocast():
                output = gen(input_photo)
                output = guided_filter(input_photo, output, r = 1)
                
                blur_fake = guided_filter(output, output, r = 5, eps = 0.2)
                blur_cartoon = guided_filter(input_cartoon, input_cartoon, r = 5, eps = 0.2)
                
                gray_fake, gray_cartoon = color_shift(output, input_cartoon)
                
                _, g_loss_gray = loss.lsgan_loss(disc_tx, gray_cartoon, gray_fake)
                _, g_loss_blur = loss.lsgan_loss(disc_sf, blur_cartoon, blur_fake)
                
                loss_content = loss.vggloss_4_4(input_photo, output)
                loss_variant = loss.total_variation_loss(output, k_size= 2)
                
                g_loss_total = g_loss_blur + loss_content + loss_variant
                # print(g_loss_total)
                
                opt_gen.zero_grad();    g_loss_total.backward(retain_graph = True)
                opt_gen.step()
                
                # opt_gen.zero_grad()
                # g_scaler.scale(g_loss_total).backward()
                # g_scaler.step(opt_gen)
                # g_scaler.update()
            
            disc_sf.train()
            disc_tx.train()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx} / {len(train_loader)}")
                print(f"\tLoss Disc Surface: {d_loss_blur:.4f},\tLoss Disc Texture: {d_loss_gray:.4f},\t Loss Gen: {g_loss_total:.4f}")
        
        if (SAVE_MODEL and (epoch + 1) % 10 == 0):
            if (not os.path.exists(CHECKPOINT_GEN)):
                os.system('mkdir -p ' + CHECKPOINT_DIR)
            
            save_checkpoint(gen, opt_gen, CHECKPOINT_GEN)
            save_checkpoint(disc_tx, opt_disc_tx, CHECKPOINT_DISC_TX)
            save_checkpoint(disc_sf, opt_disc_sf, CHECKPOINT_DISC_SF)
        
        gen.eval()
        x, y = next(val_iterator)
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            y_fake = gen(x)
            y_fake = y_fake * 0.5 + 0.5   # remove normalization #
            
        save_some_examples(x * 0.5 + 0.5, y_fake, epoch, 'evaluate')
        
        gen.train()