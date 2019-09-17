import torch
import torch.nn.functional as F


def VAE_Loss(images, re_img_mean, re_img_logvar, z_mean, z_logvar):
    # bin_size = 1. / 256.
    #
    # scale = torch.exp(re_img_logvar)
    # x = (torch.floor(images / bin_size) * bin_size - re_img_mean) / scale
    # cdf_plus = torch.sigmoid(x + bin_size / scale)
    # cdf_minus = torch.sigmoid(x)
    #
    # RE = torch.sum(torch.log(cdf_plus - cdf_minus + 1.e-7), dim=[1,2,3])

    s = 0.5 * torch.pow(re_img_mean - images, 2) / torch.exp(re_img_logvar)
    RE = torch.sum(s, dim=[1,2,3])
    # BCE = torch.sum(F.binary_cross_entropy(re_img_mean,images,reduction='none'),dim=[1,2,3])
    s = 1 + z_logvar - torch.pow(z_mean, 2) - torch.exp(z_logvar)
    KLD = 0.5 * torch.sum(s, dim=[1,2,3])
    return RE - KLD
