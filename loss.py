import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

VGG_MEAN = [104.5, 111.7, 97.72]
vgg_model = None

class Vgg19:
    def __init__(self):
        model_dict = torchvision.models.vgg19(weights = 'DEFAULT').state_dict()
    
        print('Finish loading vgg19')
    
        layers = [
            nn.Conv2d(3, 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False),
            
            nn.Conv2d(64, 128, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
        ]
        
        for i, layer in enumerate(layers):
            if (isinstance(layer, nn.Conv2d)):
                layer.weight = nn.Parameter(model_dict[f'features.{i}.weight'])
                layer.bias = nn.Parameter(model_dict[f'features.{i}.bias'])
                layer.requires_grad_ = False
        
        self.vgg_model = nn.Sequential(*layers)
        self.vgg_model.eval()
    
    def forward(self, img):
        img_scaled = (img + 1) * 127.5
        
        img[:, 0, :, :] = img_scaled[:, 2, :, :] - VGG_MEAN[0]
        img[:, 1, :, :] = img_scaled[:, 1, :, :] - VGG_MEAN[1]
        img[:, 2, :, :] = img_scaled[:, 0, :, :] - VGG_MEAN[2]
        
        return self.vgg_model(img)

def vggloss_4_4(image_a, image_b):
    global vgg_model
    if (vgg_model is None):
        vgg_model = Vgg19()
    
    vgg_a = vgg_model.forward(image_a)
    vgg_b = vgg_model.forward(image_b)
    
    return nn.MSELoss()(vgg_a, vgg_b)

# def wgan_loss(discriminator, real, fake):
#     real_logits = discriminator(real)
#     fake_logits = discriminator(fake.detach())
    
#     d_loss_real = -torch.mean(real_logits)
#     d_loss_fake = torch.mean(fake_logits)
    
#     d_loss = d_loss_real + d_loss_fake
#     g_loss = -d_loss_fake

#     """ Gradient Penalty """
#     # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
#     alpha = tf.random_uniform([tf.shape(real)[0], 1, 1, 1], minval=0.,maxval=1.)
#     differences = fake - real # This is different from MAGAN
#     interpolates = real + (alpha * differences)
#     inter_logit = discriminator(interpolates, channel=channel, name=name, reuse=True)
#     gradients = tf.gradients(inter_logit, [interpolates])[0]
#     slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
#     gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
#     d_loss += lambda_ * gradient_penalty
    
#     return d_loss, g_loss


def gan_loss(discriminator, real, fake):
    real_logit = discriminator(real)
    fake_logit = discriminator(fake.detach())
    
    real_logit = F.sigmoid(real_logit)
    fake_logit = F.sigmoid(fake_logit)
    
    g_loss = -torch.mean(torch.log(fake_logit))
    d_loss = -torch.mean(torch.log(real_logit) + torch.log(1. - fake_logit))
    
    return d_loss, g_loss

def lsgan_loss(discriminator, real, fake):
    disc_real = discriminator(real)
    disc_fake = discriminator(fake.detach())
    
    g_loss = torch.mean((disc_fake - 1)**2)
    d_loss = (torch.mean((disc_real - 1)**2) + torch.mean(disc_fake**2)) / 2
    
    return d_loss, g_loss

def total_variation_loss(image, k_size = 1):
    tv_H = torch.mean((image[:, :, k_size:, :] - image[:, :, :-k_size, :])**2)
    tv_W = torch.mean((image[:, :, :, k_size:] - image[:, :, :, :-k_size])**2)
    
    return (tv_H + tv_W) / 2

if __name__ == '__main__':
    a = torch.randn(1, 3, 256, 256)
    b = torch.randn(1, 3, 256, 256)
    
    print(vggloss_4_4(a, b))
    
    # from model import Discriminator
    
    # disc = Discriminator(channels = 3, features = 32, patch = True)
    
    # print(gan_loss(disc, a, b))
    # print(lsgan_loss(disc, a, b))