import torch
import torch.nn as nn
import torchvision
import config

VGG_MEAN = [104.5, 111.7, 97.72]
vgg_model = None

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = torchvision.models.vgg19(weights = 'DEFAULT').features[:28].eval().to(config.DEVICE)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False
        
        print("Finish loading vgg19")
    
    def _vggTransform(self, img):
        img = (torch.flip(img, dims = [1]) + 1) * 127.5
        img[:, 0, :, :] -= VGG_MEAN[0]
        img[:, 1, :, :] -= VGG_MEAN[1]
        img[:, 2, :, :] -= VGG_MEAN[2]

        return img

    def forward(self, a, b):
        vgg_a = self.vgg(self._vggTransform(a))
        vgg_b = self.vgg(self._vggTransform(b))

        return self.loss(vgg_a, vgg_b)

class GANLoss():
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.

        Note: Do not use sigmoid as the last layer of Discriminator in case using LSGAN or WGANGP
        """
        self.gan_mode = gan_mode
        # self.loss = None
        
        if (gan_mode not in ['lsgan', 'vanilla', 'wgangp']):
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """

        if (self.gan_mode == 'lsgan'):
            return torch.mean((prediction - (1.0 if target_is_real else 0.0))**2)

        if (self.gan_mode == 'wgangp'):
            return torch.mean(prediction) * (-1. if target_is_real else 1.)
        
        if (self.gan_mode == 'vanilla'):
            return -torch.mean(torch.log(prediction if target_is_real else (1. - prediction)))


def cal_lossD(discriminator, real_imgs, fake_imgs, criterion):
    pred_fake = discriminator(fake_imgs.detach())
    pred_real = discriminator(real_imgs)
    
    loss_D_fake = criterion(pred_fake, False)
    loss_D_real = criterion(pred_real, True)
    # combine loss and calculate gradients
    return (loss_D_fake + loss_D_real) * 0.5


def cal_lossG(discriminator, real_imgs, fake_imgs, criterion):
    pred_fake = discriminator(fake_imgs)
    loss_G_gan = criterion(pred_fake, True)
    
    return loss_G_gan


def total_variation_loss(image, k_size = 1):
    tv_H = torch.mean((image[:, :, k_size:, :] - image[:, :, :-k_size, :])**2)
    tv_W = torch.mean((image[:, :, :, k_size:] - image[:, :, :, :-k_size])**2)
    
    return (tv_H + tv_W) / 2


if __name__ == '__main__':
    a = torch.randn(1, 3, 256, 256).to(config.DEVICE)
    b = torch.randn(1, 3, 256, 256).to(config.DEVICE)

    print(torch.mean((a-1.0)**2))
    print(torch.mean(b**2))

    # print(torch.min(a), torch.max(a))
    # print(torch.min(b), torch.max(b))

    vggloss = VGGLoss()
    ganloss = GANLoss('lsgan')
    
    print(vggloss(a, b))
    print(ganloss(a, True))
    print(ganloss(b, False))
