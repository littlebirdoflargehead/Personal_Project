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


def Z_space_KL_Loss(z1_mean,z1_logvar,z2_mean,z2_logvar):
    '''
    计算隐变量空间分布q1(z)与q2(z)的KL散度：KL(q1||q2)
    '''
    z1_mean = z1_mean.view(-1,z1_mean.shape[1:].numel())
    z1_logvar = z1_logvar.view_as(z1_mean)
    z2_mean = z2_mean.view(-1,z2_mean.shape[1:].numel())
    z2_logvar = z2_logvar.view_as(z2_mean)

    batch_size = z2_mean.shape[0]
    kl = torch.zeros(batch_size,batch_size)
    for i in range(batch_size):
        z2_mean_temp = z2_mean[i].expand_as(z1_mean)
        z2_logvar_temp = z2_logvar[i].expand_as(z1_logvar)
        s = z1_logvar - z2_logvar_temp
        kl[i,:] = -0.5*(1+s-(z2_mean_temp-z1_mean).pow(2)/z2_logvar.exp()-s.exp()).sum(dim=1)

    return kl
