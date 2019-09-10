import torch
import torchvision
import matplotlib.pyplot as plt


def ImageVsReImagePlot(images,re_images,Config):
    '''
    将原图与使用vae重建的图进行对比
    '''
    images = images[:int(Config.total_images/2)].view(-1, 1, 28, 28)
    re_images = re_images[:int(Config.total_images/2)].view(-1, 1, 28, 28)
    images_concat = torch.cat([images, re_images], dim=3)
    images_concat = torchvision.utils.make_grid(images_concat,nrow=int(Config.image_per_row/2)).permute(1,2,0).detach().cpu()
    plt.imshow(images_concat)
    plt.show()
    return images_concat


def GenerativePlot(model,Config,random=True):
    '''
    生成VAE中decoder生成的图片
    '''
    total_images = Config.total_images*4
    image_per_row = Config.image_per_row*2
    z_dim = model.fc3.in_features

    if random:
        z = torch.randn(total_images,z_dim)
    else:
        z = torch.randn(1, z_dim)
        z = z.expand(total_images, -1).clone()
        Range = torch.linspace(-10, 10, image_per_row)
        for i in range(image_per_row):
            for j in range(image_per_row):
                z[i * image_per_row + j, 0] = Range[i]
                z[i * image_per_row + j, 1] = Range[j]

    if Config.use_gpu:
        z = z.to(Config.device)

    generative_images = model.decoder(z)
    generative_images = generative_images.view(-1, 1, 28, 28)
    generative_images = torchvision.utils.make_grid(generative_images, nrow=image_per_row).permute(1, 2, 0).detach().cpu()
    plt.imshow(generative_images)
    plt.show()
    return generative_images
