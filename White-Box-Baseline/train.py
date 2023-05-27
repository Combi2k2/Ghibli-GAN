import os
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import config
import model
import loss

from filters import guided_filter, color_shift, simple_superpixel
from utils import save_some_examples, save_checkpoint, load_checkpoint
from dataset import GhibliDataset, AnimeFaceDataset

if __name__ == '__main__':
    # Initialize the models
    gen = model.Unet_Generator(channels = config.CHANNELS_IMG, features = config.FEATURES_GEN, num_blocks = 4).to(config.DEVICE)
    disc_sf = model.Discriminator(channels = config.CHANNELS_IMG, features = config.FEATURES_DISC, patch = True).to(config.DEVICE)
    disc_tx = model.Discriminator(channels = 1, features = config.FEATURES_DISC, patch = True).to(config.DEVICE)
    
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    opt_gen = optim.Adam(gen.parameters(), lr = config.LEARNING_RATE, betas = config.BETAS)
    opt_disc_sf = optim.Adam(disc_sf.parameters(), lr = config.LEARNING_RATE, betas = config.BETAS)
    opt_disc_tx = optim.Adam(disc_tx.parameters(), lr = config.LEARNING_RATE, betas = config.BETAS)

    if not os.path.exists(config.CHECKPOINT_DIR):
        os.system(f'mkdir {config.CHECKPOINT_DIR}')

    if config.LOAD_CHECKPOINT == True:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_TX, disc_tx, opt_disc_tx, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_SF, disc_sf, opt_disc_sf, config.LEARNING_RATE)

    print("Finish initializing model")

    # Initialize the dataset
    dataset = AnimeFaceDataset()#'data/Real-Images', 'data/Cartoon', 1000)

    train_ds, valid_ds = train_test_split(dataset, test_size = 0.1, shuffle = True)

    train_loader = DataLoader(train_ds, batch_size = config.BATCH_SIZE, shuffle = True, num_workers = config.NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, batch_size = config.BATCH_SIZE, num_workers = config.NUM_WORKERS)

    val_iterator = iter(valid_loader)

    print("Finish initializing dataset")

    # Start training
    torch.autograd.set_detect_anomaly(True)

    vggloss = loss.VGGLoss().to(config.DEVICE)
    ganloss = loss.GANLoss('lsgan', target_real_label=1.0, target_fake_label=0.0).to(config.DEVICE)
    
    for epoch in range(config.NUM_EPOCHS):
        for batch_idx, (input_photo, input_cartoon) in enumerate(train_loader):
            input_superpixel = torch.from_numpy(simple_superpixel(input_photo.permute(0, 2, 3, 1).numpy())).permute(0, 3, 1, 2).to(config.DEVICE)
            input_cartoon = input_cartoon.to(config.DEVICE)
            input_photo = input_photo.to(config.DEVICE)
            
            # generate the output and other feature maps
            output = gen(input_photo)
            output = guided_filter(input_photo, output, r = 1)
            
            blur_fake = guided_filter(output, output, r = 5, eps = 0.2)
            blur_cartoon = guided_filter(input_cartoon, input_cartoon, r = 5, eps = 0.2)
            
            gray_fake, gray_cartoon = color_shift(output, input_cartoon)

            # Train Discriminator
            for p in list(disc_sf.parameters()) + list(disc_tx.parameters()):
                p.requires_grad_(True)

            with torch.cuda.amp.autocast():
                d_loss_gray = loss.cal_lossD(disc_tx, gray_cartoon, gray_fake, ganloss)
                d_loss_blur = loss.cal_lossD(disc_sf, blur_cartoon, blur_fake, ganloss)

                opt_disc_tx.zero_grad(set_to_none = True);  d_loss_gray.backward()
                opt_disc_tx.step()

                opt_disc_sf.zero_grad(set_to_none = True);  d_loss_blur.backward()
                opt_disc_sf.step()
                
                # disc_tx.zero_grad(set_to_none = True)
                # d_scaler.scale(d_loss_gray).backward()
                # d_scaler.step(opt_disc_tx)
                # d_scaler.update()
                
                # disc_sf.zero_grad(set_to_none = True)
                # d_scaler.scale(d_loss_blur).backward()
                # d_scaler.step(opt_disc_sf)
                # d_scaler.update()
            
            # Train generator
            for p in list(disc_sf.parameters()) + list(disc_tx.parameters()):
                p.requires_grad_(False)
            
            with torch.cuda.amp.autocast():
                g_loss_gray = loss.cal_lossG(disc_tx, gray_cartoon, gray_fake, ganloss)
                g_loss_blur = loss.cal_lossG(disc_sf, blur_cartoon, blur_fake, ganloss)

                loss_content = vggloss(output, input_photo)
                loss_structure = vggloss(output, input_superpixel)

                loss_variant = loss.total_variation_loss(output)

                g_loss_total = config.LAMBDA_TEXTURE * g_loss_gray + \
                            config.LAMBDA_SURFACE * g_loss_blur + \
                            config.LAMBDA_CONTENT * (loss_content + loss_structure) + \
                            config.LAMBDA_VARIANT * loss_variant

                opt_gen.zero_grad(set_to_none = True); g_loss_total.backward()
                opt_gen.step()

                # opt_gen.zero_grad()
                # g_scaler.scale(g_loss_total).backward()
                # g_scaler.step(opt_gen)
                # g_scaler.update()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{config.NUM_EPOCHS}] Batch {batch_idx} / {len(train_loader)}")
                print(f"\tLoss Disc: (surface: {d_loss_blur:.4f}, texture: {d_loss_gray:.4f}),\t Loss Gen: (surface: {g_loss_gray:.4f}, texture: {g_loss_blur:.4f}, content: {loss_content:.4f}, structure: {loss_structure:.4f}, variant: {loss_variant:.4f}")
        
        if (config.SAVE_MODEL and (epoch + 1) % 10 == 0):
            if (not os.path.exists(config.CHECKPOINT_DIR)):
                os.system('mkdir -p ' + config.CHECKPOINT_DIR)
            
            save_checkpoint(gen, opt_gen, config.CHECKPOINT_GEN)
            save_checkpoint(disc_tx, opt_disc_tx, config.CHECKPOINT_DISC_TX)
            save_checkpoint(disc_sf, opt_disc_sf, config.CHECKPOINT_DISC_SF)
        
        gen.eval()
        
        try:
            x, _ = next(val_iterator)
        except:
            val_iterator = iter(valid_loader)
            x, _ = next(val_iterator)
        
        x = x.to(config.DEVICE)
        
        with torch.no_grad():
            y_fake = gen(x)
            y_fake = y_fake * 0.5 + 0.5   # remove normalization #
            
        save_some_examples(x * 0.5 + 0.5, y_fake, epoch, 'evaluate')
        
        gen.train()