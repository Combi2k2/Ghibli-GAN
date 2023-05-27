import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

VGG_MEAN = [103.939, 116.779, 123.68]
vgg_model = None

class Vgg19:
    def __init__(self):
        super().__init__()
        self.vgg = torchvision.models.vgg19(weights = 'DEFAULT').features[:28].eval()
        self.loss = nn.L1Loss()

        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.vgg_model = nn.Sequential(*layers)
        self.vgg_model.eval()
    
    def _vggTransform(self, img):
        img = (torch.flip(img, dims = [1]) + 1) * 127.5
        img[:, 0, :, :] -= VGG_MEAN[0]
        img[:, 1, :, :] -= VGG_MEAN[1]
        img[:, 2, :, :] -= VGG_MEAN[2]

        return img / 127.5 - 1

    def forward(self, a, b):
        vgg_a = self.vgg(self._vggTransform(a))
        vgg_b = self.vgg(self._vggTransform(b))

        return self.loss(vgg_a, vgg_b)

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_lossD(discriminator, real_imgs, fake_imgs, gan_loss):
    pred_fake = discriminator(fake_imgs.detach())
    pred_real = discriminator(real_imgs)
    
    loss_D_fake = gan_loss(pred_fake, False)
    loss_D_real = gan_loss(pred_real, True)
    # combine loss and calculate gradients
    return (loss_D_fake + loss_D_real) * 0.5

def cal_lossG(discriminator, real_imgs, fake_imgs, gan_loss):
    pred_fake = discriminator(fake_imgs)
    loss_G_gan = gan_loss(pred_fake, True)
    
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
    a = torch.randn(1, 3, 256, 256).to(config.DEVICE)
    b = torch.randn(1, 3, 256, 256).to(config.DEVICE)

    print(torch.mean((a-1.0)**2))
    print(torch.mean(b**2))

    # print(torch.min(a), torch.max(a))
    # print(torch.min(b), torch.max(b))

    vggloss = VGGLoss().to(config.DEVICE)
    ganloss = GANLoss('lsgan').to(config.DEVICE)

    print(vggloss.vgg(a).shape)
    
    # print(vggloss(a, b))
    # print(ganloss(a, True))
    # print(ganloss(b, False))
