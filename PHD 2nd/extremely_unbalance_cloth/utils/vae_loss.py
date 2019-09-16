import torch
import torch.nn.functional as F


def VAE_Loss(images, re_img_mean, re_img_logvar, z_mean, z_logvar):
    s = 0.5 * (re_img_logvar + torch.pow(re_img_mean - images, 2) / torch.exp(re_img_logvar))
    RE = torch.sum(s, dim=[1,2,3])
    # BCE = torch.sum(F.binary_cross_entropy(re_images,images,reduction='none'),dim=[1,2,3])
    s = 1 + z_logvar - torch.pow(z_mean, 2) - torch.exp(z_logvar)
    KLD = 0.5 * torch.sum(s, dim=1)
    return RE - KLD
