import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_nc, feat_dim = 528, features = 64, n_downsampling = 2, bias = True, act = "relu"):
        """Construct an Encoder
        Parameters:
            input_nc            -- the number of channels in input images
            feat_dim            -- the number of channels in the encoded output
            features            -- the number of filters in the last conv layer
            n_downsampling      -- the number of downsample times
            act                 -- choice of activation layer ('relu' or 'leaky')
        """  
        super().__init__()
        
        moduleList = [
            nn.Conv2d(input_nc, features, kernel_size = 7, padding = 3, bias = bias, padding_mode = 'reflect'),
            nn.InstanceNorm2d(features),
            nn.ReLU(True) if act == "relu" else nn.LeakyReLU(0.2)
        ]
        for i in range(n_downsampling):
            in_channels = features << i
            out_channels = in_channels * 2 if i < n_downsampling - 1 else feat_dim
            
            moduleList += [
                nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias = bias, padding_mode = 'reflect'),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True) if act == "relu" else nn.LeakyReLU(0.2)
            ]
        
        self.model = nn.Sequential(*moduleList)
    
    def forward(self, inputs):
        return self.model(inputs)

class Decoder(nn.Module):
    def __init__(self, output_nc, feat_dim = 528, features = 64, n_downsampling = 2, bias = True, act = "relu"):
        """Construct a Decoder
        Parameters:
            output_nc           -- the number of channels in output image
            feat_dim            -- the number of channels in the decoding input
            features            -- the number of filters in the last conv layer
            n_downsampling      -- the number of upsample times
            act                 -- choice of activation layer ('relu' or 'leaky')
        """  
        super().__init__()
        
        moduleList = []
        
        for i in range(n_downsampling):
            input_chan = features << (n_downsampling - i) if i > 0 else feat_dim
            output_chan = features << (n_downsampling - i - 1)
            
            moduleList += [
                nn.ConvTranspose2d(input_chan, output_chan, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                nn.BatchNorm2d(output_chan),
                nn.ReLU(inplace = True)
            ]

        moduleList.append(nn.ReflectionPad2d(3))
        moduleList.append(nn.Conv2d(features, output_nc, kernel_size = 7, padding = 0))
        moduleList.append(nn.ReLU(inplace = True))

        self.model = nn.Sequential(*moduleList)
    
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    enc = Encoder(3, 128)
    dec = Decoder(3, 128)
    
    inputs = torch.randn(3, 3, 256, 256)
    output = dec(enc(inputs))
    
    print(output.shape)