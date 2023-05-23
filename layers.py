import torch
import torch.nn as nn
import torch.nn.functional as F

# def adaptive_instance_norm(content, style, epsilon=1e-5):
#     c_mean, c_var = torch.mean(content, dim = (1, 2), keepdim = True), \
#                     torch.var(content, dim = (1, 2), keepdim = True)
#     s_mean, s_var = torch.mean(content, dim = (1, 2), keepdim = True), \
#                     torch.var(content, dim = (1, 2), keepdim = True)
#     c_std, s_std = torch.sqrt(c_var + epsilon), torch.sqrt(s_var + epsilon)

#     return s_std * (content - c_mean) / c_std + s_mean

# def adaptive_instance_norm_tf(content, style, epsilon=1e-5):
#     c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
#     s_mean, s_var = tf.nn.moments(style, axes=[1, 2], keep_dims=True)
#     c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)

#     return s_std * (content - c_mean) / c_std + s_mean

def spectral_norm(w, iteration = 1):
    w_shape = list(w.shape)
    w = torch.reshape(w, (-1, w_shape[-1]))

    u = torch.empty((1, w_shape[-1])).normal_(mean = 0, std = 1)
    
    u_hat = u
    v_hat = None
    
    for _ in range(iteration):
        v_ = torch.matmul(u_hat, torch.transpose(w, 0, 1))
        v_hat = F.normalize(v_)
        
        u_ = torch.matmul(v_hat, w)
        u_hat = F.normalize(u_)

    u_hat = u_hat.detach()
    v_hat = v_hat.detach()

    sigma = torch.matmul(torch.matmul(v_hat, w), torch.transpose(u_hat, 0, 1))

    u = u_hat
    w_norm = w / sigma
    w_norm = torch.reshape(w_norm, w_shape)

    return w_norm

def conv_spectral_norm(in_channels, out_channels, kernel_size, stride):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = (kernel_size - 1) // 2)
    
    layer.weight = nn.Parameter(spectral_norm(layer.weight))
    layer.bias = nn.Parameter(torch.zeros((out_channels,)))
    
    return layer
    
if __name__ == '__main__':
    filter = conv_spectral_norm(3, 32, 3, 2)
    
    inputs = torch.randn(1, 3, 32, 32)
    output = filter(inputs)
    
    print(output.shape)
    # torch.Size([1, 32, 16, 16])