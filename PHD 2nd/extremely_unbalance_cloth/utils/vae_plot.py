import torch
import torchvision
import matplotlib.pyplot as plt


def ImageVsReImagePlot(images,re_images,Config):
    '''
    将原图与使用vae重建的图进行对比
    '''
    images = images[:int(Config.total_images/2)]
    re_images = re_images[:int(Config.total_images/2)]
    images_concat = torch.cat([images, re_images], dim=3)
    # torchvision.utils.save_image(images_concat, 'ImageVsReImagePlot.png',nrow=int(Config.image_per_row/2))
    images_concat = torchvision.utils.make_grid(images_concat,nrow=int(Config.image_per_row/2)).permute(1,2,0).detach().cpu()
    plt.imshow(images_concat)
    plt.show()
    return images_concat


def GenerativePlot(model,Config,random=True):
    '''
    生成VAE中decoder生成的图片
    '''
    total_images = Config.total_images
    image_per_row = Config.image_per_row
    z_dim = model.z_dim

    if random:
        z = torch.randn(total_images,z_dim[0],z_dim[1],z_dim[2])
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

    generative_images,_ = model.decoder(z)
    # torchvision.utils.save_image(generative_images, 'GenerativePlot.png', nrow=image_per_row)
    generative_images = torchvision.utils.make_grid(generative_images, nrow=image_per_row).permute(1, 2, 0).detach().cpu()
    plt.imshow(generative_images)
    plt.show()
    return generative_images
