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

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, act = "relu", bias = True, use_dropout = False):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = kernel_size, stride = 2, padding = (kernel_size - 1) // 2, bias = bias),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = (kernel_size - 1) // 2, bias = False),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.dropout(out) if self.use_dropout else out
        
        return out

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, act = "relu", bias = True, use_dropout = False):
        super().__init__()

        self.up = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = kernel_size, padding = "same", bias = bias),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = "same", bias = bias),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x_small, x_large):
        out = self.up(x_small)
        out = self.conv(x_large + out)
        out = self.dropout(out) if self.use_dropout else out
        
        return out


class Unet_Generator(nn.Module):
    def __init__(self, channels = 3, features = 32, num_blocks = 4):
        super().__init__()
        
        self.initial_layer = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size = 7, stride = 1, padding = 3),
            nn.LeakyReLU(0.2)
        )
        self.down1 = Down(features, features * 2, act = "leaky")
        self.down2 = Down(features * 2, features * 4, act = "leaky")
        
        self.bottle_neck = nn.ModuleList([ResnetBlock(features * 4, features * 4) for _ in range(num_blocks)])
        self.bottle_neck = nn.Sequential(
            *self.bottle_neck,
            nn.Conv2d(features * 4, features * 2, kernel_size = 3, padding = 1),
            nn.LeakyReLU(0.2)
        )

        self.up1 = Up(features * 2, features, act = "leaky")
        self.up2 = Up(features, channels, act = "leaky")

        self.final_layer = nn.Sequential(
            nn.Identity(),
            # nn.Conv2d(channels, channels, kernel_size = 7, padding = 3)
        )
    
    def forward(self, x0):
        x0 = self.initial_layer(x0)
        x1 = self.down1(x0)
        x2 = self.down2(x1)

        x2 = self.bottle_neck(x2)

        x3 = self.up1(x2, x1)
        x4 = self.up2(x3, x0)

        return self.final_layer(x4)

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