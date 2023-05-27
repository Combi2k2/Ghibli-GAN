from skimage import segmentation, color
from joblib import Parallel, delayed

import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def box_filter(x, r):
    channels = x.shape[1]
    
    weight = 1 / ((2 * r + 1)**2)
    kernel = torch.ones((channels, 1, 2*r+1, 2*r+1), dtype = x.dtype, device = device) / weight
    
    return F.conv2d(x, kernel, padding = r, groups = channels)

def guided_filter(x, y, r, eps = 1e-2):
    '''
        Input:
            x, y:   tensor or np.ndarray of format N x C x H x W
            r:      an interger
        Return:
            An extracted surface representation with textures and details removed
            The extracted representation has the same shape with the inputs

    '''
    _, _, H, W = x.shape
    
    N = box_filter(torch.ones((1, 1, H, W), dtype = x.dtype, device = device), r)
    
    mean_x = box_filter(x, r) / N
    mean_y = box_filter(y, r) / N
    cov_xy = box_filter(x * y, r) / N - mean_x * mean_y
    var_x  = box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = box_filter(A, r) / N
    mean_b = box_filter(b, r) / N
    
    return mean_A * x + mean_b

def color_shift(image1, image2, mode='uniform'):
    b1, g1, r1 = torch.split(image1, [1, 1, 1], dim = 1)
    b2, g2, r2 = torch.split(image2, [1, 1, 1], dim = 1)
    
    if mode == 'normal':
        b_weight = np.random.normal(0.114, 0.1)
        g_weight = np.random.normal(0.587, 0.1)
        r_weight = np.random.normal(0.299, 0.1)
    elif mode == 'uniform':
        b_weight = np.random.uniform(0.014, 0.214)
        g_weight = np.random.uniform(0.487, 0.687)
        r_weight = np.random.uniform(0.199, 0.399)
    
    output1 = (b_weight * b1 + g_weight * g1 + r_weight * r1) / (b_weight + g_weight + r_weight)
    output2 = (b_weight * b2 + g_weight * g2 + r_weight * r2) / (b_weight + g_weight + r_weight)
    
    return output1, output2


def label2rgb(label_field, image, kind = 'mix', bg_label = -1, bg_color = (0, 0, 0)):
    labels = np.unique(label_field)

    out = np.zeros_like(image)
    bg = (labels == bg_label)

    if bg.any():
        labels = labels[labels != bg_label]
        mask = (label_field == bg_label).nonzero()
        out[mask] = bg_color
    for label in labels:
        mask = (label_field == label).nonzero()
        #std = np.std(image[mask])
        #std_list.append(std)
        if kind == 'avg':
            color = image[mask].mean(axis=0)
        elif kind == 'median':
            color = np.median(image[mask], axis=0)
        elif kind == 'mix':
            std = np.std(image[mask])
            if std < 20:
                color = image[mask].mean(axis=0)
            elif 20 < std < 40:
                mean = image[mask].mean(axis=0)
                median = np.median(image[mask], axis=0)
                color = 0.5*mean + 0.5*median
            elif 40 < std:
                color = np.median(image[mask], axis=0)
        out[mask] = color
    return out

def simple_superpixel(batch_image, seg_num = 200):
    def process_slic(image):
        seg_label = segmentation.slic(image, n_segments=seg_num, sigma=1,
                                        compactness=10, convert2lab=True)
        image = color.label2rgb(seg_label, image, kind='avg')
        return image
    
    num_job = np.shape(batch_image)[0]
    batch_out = Parallel(n_jobs=num_job)(delayed(process_slic)(image) for image in batch_image)

    return np.array(batch_out)

if __name__ == '__main__':
    # x = torch.randn((16, 3, 64, 64)).to(device)
    # y = torch.randn((16, 3, 64, 64)).to(device)
    
    # print(guided_filter(x, y, 2).shape)

    import cv2

    img = cv2.imread('data/Real-Images/Room/Image135.jpg')
    img = np.expand_dims(img, axis = 0) / 127.5 - 1

    img_superpixel = simple_superpixel(img, seg_num = 500)
    img_superpixel = (img_superpixel[0] + 1) * 127.5

    cv2.imwrite('superpixel.png', img_superpixel)