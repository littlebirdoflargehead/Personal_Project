import torch
import torch.nn.functional as F


def VAE_Loss(images,re_images,mu,logvar):
    BCE = torch.sum(F.binary_cross_entropy(re_images,images,reduction='none'),dim=[1,2,3])
    s = 1+logvar-torch.pow(mu,2)-torch.exp(logvar)
    KLD = 0.5*torch.sum(s,dim=1)
    return BCE - KLD