import torch
import torch.nn.functional as F


def VAE_Loss(images,re_images,mu,logvar):
    BCE = F.binary_cross_entropy(re_images,images,reduction='mean')
    s = 1+logvar-torch.pow(mu,2)-torch.exp(logvar)
    KLD = 0.5*torch.mean(s)
    return BCE - KLD