from __future__ import unicode_literals, print_function, division

import torch.nn as nn
import torch

import numpy as np
import itertools

import cv2
import os
import gc

from utils import tensor2img, tensor2img_with_heatmap
from utils import load_checkpoint, save_checkpoint
from models import Generator, Discriminator

import config
import loss

def cal_lossD(discriminator, real_imgs, fake_imgs, gan_loss):
    fake_logit, fake_cam_logit, _ = discriminator(fake_imgs)
    real_logit, real_cam_logit, _ = discriminator(real_imgs)
    
    loss_D_fake = gan_loss(fake_logit, False) + gan_loss(fake_cam_logit, False)
    loss_D_real = gan_loss(real_logit, True) + gan_loss(real_cam_logit, True)
    # combine loss and calculate gradients
    
    return loss_D_fake + loss_D_real

def cal_lossG(discriminator, fake_imgs, gan_loss):
    pred_fake, pred_fake_cam, _ = discriminator(fake_imgs)
    
    loss_G_adv = gan_loss(pred_fake, True)
    loss_G_adv_cam = gan_loss(pred_fake_cam, True)
    
    return loss_G_adv + loss_G_adv_cam

class Trainer(object):
    def __init__(self):
        self.build_model()
        
        if (config.LOAD_CHECKPOINT):
            self.load()
        
        # Define loss function
        self.criterion_recon = nn.L1Loss().to(config.DEVICE)
        self.criterion_CAM = loss.GANLoss('vanilla').to(config.DEVICE)
        self.criterion_GAN = loss.GANLoss('lsgan').to(config.DEVICE)
    
    def build_model(self):
        self.genA2B = Generator(config.CHANNELS_IMG, config.FEATURES_GEN, config.EMBED_DIM, config.N_DOWNSAMPLING, config.N_RESTNET_BLOCK).to(config.DEVICE)
        self.genB2A = Generator(config.CHANNELS_IMG, config.FEATURES_GEN, config.EMBED_DIM, config.N_DOWNSAMPLING, config.N_RESTNET_BLOCK).to(config.DEVICE)
        self.discA = Discriminator(config.CHANNELS_IMG, config.FEATURES_DISC, config.EMBED_DIM, config.N_DOWNSAMPLING).to(config.DEVICE)
        self.discB = Discriminator(config.CHANNELS_IMG, config.FEATURES_DISC, config.EMBED_DIM, config.N_DOWNSAMPLING).to(config.DEVICE)
        
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr = config.LEARNING_RATE, betas = config.BETA, weight_decay = config.WEIGHT_DECAY)
        self.D_optim = torch.optim.Adam(itertools.chain(self.discA.parameters(), self.discB.parameters()), lr = config.LEARNING_RATE, betas = config.BETA, weight_decay = config.WEIGHT_DECAY)
    
    def run_train(self, sample):
        gc.collect()
        torch.cuda.empty_cache()
        
        real_A = sample[0].to(config.DEVICE)
        real_B = sample[1].to(config.DEVICE)
        
        fake_A2B, _, _ = self.genA2B(real_A)
        fake_B2A, _, _ = self.genB2A(real_B)
        
        # Train discriminator
        D_loss_A = config.ADV_WEIGHT * cal_lossD(self.discA, real_A, fake_B2A, self.criterion_GAN)
        D_loss_B = config.ADV_WEIGHT * cal_lossD(self.discB, real_B, fake_A2B, self.criterion_GAN)
        
        self.D_optim.zero_grad()
        Discriminator_loss = D_loss_A + D_loss_B
        Discriminator_loss.backward()
        self.D_optim.step()
        
        # Train Generator
        fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
        fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

        fake_A2B2A, _, _ = self.genB2A(fake_A2B)
        fake_B2A2B, _, _ = self.genA2B(fake_B2A)

        fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
        fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)
        
        # Adversarial Loss
        G_adv_loss_A = cal_lossG(self.discA, fake_B2A, self.criterion_GAN)
        G_adv_loss_B = cal_lossG(self.discB, fake_A2B, self.criterion_GAN)

        # Cycle Loss
        G_recon_loss_A = self.criterion_recon(fake_A2B2A, real_A)
        G_recon_loss_B = self.criterion_recon(fake_B2A2B, real_B)

        # Identity Loss
        G_identity_loss_A = self.criterion_recon(fake_A2A, real_A)
        G_identity_loss_B = self.criterion_recon(fake_B2B, real_B)

        G_cam_loss_A = self.criterion_CAM(fake_B2A_cam_logit, True) + self.criterion_CAM(fake_A2A_cam_logit, False)
        G_cam_loss_B = self.criterion_CAM(fake_A2B_cam_logit, True) + self.criterion_CAM(fake_B2B_cam_logit, False)

        G_loss_A = config.ADV_WEIGHT * G_adv_loss_A + config.CYCLE_WEIGHT * G_recon_loss_A + config.IDENTITY_WEIGHT * G_identity_loss_A + config.CAM_WEIGHT * G_cam_loss_A
        G_loss_B = config.ADV_WEIGHT * G_adv_loss_B + config.CYCLE_WEIGHT * G_recon_loss_B + config.IDENTITY_WEIGHT * G_identity_loss_B + config.CAM_WEIGHT * G_cam_loss_B

        self.G_optim.zero_grad()
        Generator_loss = G_loss_A + G_loss_B
        Generator_loss.backward()
        self.G_optim.step()
        
        return {
            'D_loss': Discriminator_loss,
            'G_loss': {
                'total_loss': Generator_loss,
                'adv': G_adv_loss_A + G_adv_loss_B,
                'cam': G_cam_loss_A + G_cam_loss_B,
                'cycle': G_recon_loss_A + G_recon_loss_B,
                'identity': G_identity_loss_A + G_identity_loss_B
            }
        }
    
    def run_test(self, sample, save_dir):
        real_A = sample[0].to(config.DEVICE)
        real_B = sample[1].to(config.DEVICE)
        
        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)
        
        n = real_A.shape[0]
        
        batch_imageA = None
        batch_imageB = None
        
        for i in range(n):
            A2B = np.concatenate((
                tensor2img(real_A[i]),
                tensor2img(fake_A2B[i]),
                tensor2img(fake_A2B2A[i]),
                tensor2img_with_heatmap(fake_A2A[i], fake_A2A_heatmap[i]),
                tensor2img_with_heatmap(fake_A2B[i], fake_A2B_heatmap[i]),
                tensor2img_with_heatmap(fake_A2B2A[i], fake_A2B2A_heatmap[i]),
            ), 1)
        
            B2A = np.concatenate((
                tensor2img(real_B[i]),
                tensor2img(fake_B2A[i]),
                tensor2img(fake_B2A2B[i]),
                tensor2img_with_heatmap(fake_B2B[i], fake_B2B_heatmap[i]),
                tensor2img_with_heatmap(fake_B2A[i], fake_B2A_heatmap[i]),
                tensor2img_with_heatmap(fake_B2A2B[i], fake_B2A2B_heatmap[i]),
            ), 1)
            
            if batch_imageA:
                batch_imageA = np.concatenate((batch_imageA, A2B), dim = 0)
                batch_imageB = np.concatenate((batch_imageB, B2A), dim = 0)
            else:
                batch_imageA, batch_imageB = A2B, B2A
        
        if not os.path.exists(save_dir):
            os.system(f"mkdir {save_dir}")
        
        cv2.imwrite(os.path.join(save_dir, 'imageA.png'), batch_imageA * 255.0)
        cv2.imwrite(os.path.join(save_dir, 'imageB.png'), batch_imageB * 255.0)
    
    def load(self, dir = config.CHECKPOINT_DIR):
        load_checkpoint(self.genA2B, f'{dir}/GeneratorA2B.pt')
        load_checkpoint(self.genB2A, f'{dir}/GeneratorB2A.pt')
        load_checkpoint(self.discA, f'{dir}/Discriminator2A.pt')
        load_checkpoint(self.discB, f'{dir}/Discriminator2B.pt')
        load_checkpoint(self.G_optim, f'{dir}/Optimizer_G.pt', config.LEARNING_RATE)
        load_checkpoint(self.D_optim, f'{dir}/Optimizer_D.pt', config.LEARNING_RATE)
    
    def save(self, dir = config.CHECKPOINT_DIR):
        save_checkpoint(self.genA2B, f'{dir}/GeneratorA2B.pt')
        save_checkpoint(self.genB2A, f'{dir}/GeneratorB2A.pt')
        save_checkpoint(self.discA, f'{dir}/Discriminator2A.pt')
        save_checkpoint(self.discB, f'{dir}/Discriminator2B.pt')
        save_checkpoint(self.G_optim, f'{dir}/Optimizer_G.pt')
        save_checkpoint(self.D_optim, f'{dir}/Optimizer_D.pt')
    
    def train(self):
        self.genA2B.train(), self.genB2A.train(), self.discA.train(), self.discB.train()
    
    def eval(self):
        self.genA2B.eval(), self.genB2A.eval(), self.discA.eval(), self.discB.eval()

 
if __name__ == '__main__':
    from utils import count_parameters
    
    trainer = Trainer()
    
    print(count_parameters(trainer.genA2B))
    print(count_parameters(trainer.discA))
    
    trainer.save()
    trainer.load()
    
    print("Load and Save checked")
    
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)
    
    trainer.run_train((x, y))
    trainer.run_test((x, y), 'evaluate')
    
    print("Train and Test checked")
            
            
