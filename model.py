import torch.nn as nn
import torch
import layers

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.resnet = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        )
    
    def forward(self, x):
        return self.resnet(x) + x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down = True, act = "relu", use_dropout = False):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = "same"),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        ) if down else nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True),
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = "same"),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = "same"),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)
        # x = self.conv(x)
        # return self.dropout(x) if self.use_dropout else x

class Unet_Generator(nn.Module):
    def __init__(self, channels = 3, features = 32, num_blocks = 4):
        super().__init__()
        
        self.initial_down = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size = 5, stride = 2, padding = 2),
            nn.LeakyReLU(0.2)
        )
        self.down1 = ConvBlock(features, features * 2, down = True, act = "leaky")
        self.down2 = ConvBlock(features * 2, features * 4, down = True, act = "leaky")
        self.down3 = ConvBlock(features * 4, features * 8, down = True, act = "leaky")
        
        self.bottle_neck = nn.ModuleList([ResnetBlock(features * 8, features * 8) for _ in range(num_blocks)])
        self.bottle_neck = nn.Sequential(*self.bottle_neck)
        
        self.up1 = ConvBlock(features * 8, features * 4, down = False, act = "leaky")
        self.up2 = ConvBlock(features * 8, features * 2, down = False, act = "leaky")
        self.up3 = ConvBlock(features * 4, features, down = False, act = "leaky")
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, channels, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        
        bottle_neck = self.bottle_neck(d4)
        
        up1 = self.up1(bottle_neck)
        up2 = self.up2(torch.cat([up1, d3], dim = 1))
        up3 = self.up3(torch.cat([up2, d2], dim = 1))
        
        return self.final_up(torch.cat([up3, d1], dim = 1))

class Discriminator(nn.Module):
    def __init__(self, channels = 3, features = 32, patch = True):
        super().__init__()
        self.disc = nn.ModuleList()
        self.patch = patch
        
        for i in range(3):
            in_channels = features<<(i-1) if i > 0 else channels
            out_channels = features<<i

            self.disc.append(layers.conv_spectral_norm(in_channels, out_channels, 3, 2))
            self.disc.append(nn.LeakyReLU(0.2))
            self.disc.append(layers.conv_spectral_norm(out_channels, out_channels, 3, 1))
            self.disc.append(nn.LeakyReLU(0.2))
        
        self.disc = nn.Sequential(*self.disc)
        
        
        if patch:   self.final_layer = layers.conv_spectral_norm(out_channels, 1, 1, 1)
        else:       self.final_layer = nn.Linear(out_channels, 1)
    
    def forward(self, x):
        x = self.disc(x)
        x = self.final_layer(x if self.patch else torch.mean(x, dim = (2, 3)))
        
        return x
            
# def disc_bn(x, scale=1, channel=32, is_training=True, 
#             name='discriminator', patch=True, reuse=False):
    
#     with tf.variable_scope(name, reuse=reuse):
        
#         for idx in range(3):
#             x = slim.convolution2d(x, channel*2**idx, [3, 3], stride=2, activation_fn=None)
#             x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
#             x = tf.nn.leaky_relu(x)
            
#             x = slim.convolution2d(x, channel*2**idx, [3, 3], activation_fn=None)
#             x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
#             x = tf.nn.leaky_relu(x)

#         if patch == True:
#             x = slim.convolution2d(x, 1, [1, 1], activation_fn=None)
#         else:
#             x = tf.reduce_mean(x, axis=[1, 2])
#             x = slim.fully_connected(x, 1, activation_fn=None)
        
#         return x

# def disc_sn(x, scale=1, channel=32, patch=True, name='discriminator', reuse=False):
#     with tf.variable_scope(name, reuse=reuse):

#         for idx in range(3):
#             x = layers.conv_spectral_norm(x, channel*2**idx, [3, 3], 
#                                           stride=2, name='conv{}_1'.format(idx))
#             x = tf.nn.leaky_relu(x)
            
#             x = layers.conv_spectral_norm(x, channel*2**idx, [3, 3], 
#                                           name='conv{}_2'.format(idx))
#             x = tf.nn.leaky_relu(x)
        
        
#         if patch == True:
#             x = layers.conv_spectral_norm(x, 1, [1, 1], name='conv_out'.format(idx))
            
#         else:
#             x = tf.reduce_mean(x, axis=[1, 2])
#             x = slim.fully_connected(x, 1, activation_fn=None)
        
#         return x


# def disc_ln(x, channel=32, is_training=True, name='discriminator', patch=True, reuse=False):
#     with tf.variable_scope(name, reuse=reuse):

#         for idx in range(3):
#             x = slim.convolution2d(x, channel*2**idx, [3, 3], stride=2, activation_fn=None)
#             x = tf.contrib.layers.layer_norm(x)
#             x = tf.nn.leaky_relu(x)
            
#             x = slim.convolution2d(x, channel*2**idx, [3, 3], activation_fn=None)
#             x = tf.contrib.layers.layer_norm(x)
#             x = tf.nn.leaky_relu(x)

#         if patch == True:
#             x = slim.convolution2d(x, 1, [1, 1], activation_fn=None)
#         else:
#             x = tf.reduce_mean(x, axis=[1, 2])
#             x = slim.fully_connected(x, 1, activation_fn=None)
        
#         return x

if __name__ == '__main__':
    gen = Unet_Generator(channels = 3, features = 32, num_blocks = 4)
    disc = Discriminator(channels = 3, features = 32, patch = True)
    
    inputs = torch.randn(10, 3, 256, 256)
    output_gen = gen(inputs)
    output_disc = disc(inputs)
    
    print(output_gen.shape)
    print(output_disc.shape)