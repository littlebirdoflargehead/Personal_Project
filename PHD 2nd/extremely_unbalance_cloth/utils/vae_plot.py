import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


def ImageVsReImagePlot(images, re_images, Config):
    '''
    将原图与使用vae重建的图进行对比
    '''
    images = images[:int(Config.total_images / 2)]
    re_images = re_images[:int(Config.total_images / 2)]
    images_concat = torch.cat([images, re_images], dim=3)
    # torchvision.utils.save_image(images_concat, 'ImageVsReImagePlot.png',nrow=int(Config.image_per_row/2))
    images_concat = torchvision.utils.make_grid(images_concat, nrow=int(Config.image_per_row / 2)).permute(1, 2,
                                                                                                           0).detach().cpu()
    plt.imshow(images_concat)
    plt.show()
    return images_concat


def GenerativePlot(model, Config, random=True):
    '''
    生成VAE中decoder生成的图片
    '''
    total_images = Config.total_images
    image_per_row = Config.image_per_row
    z_dim = model.z_dim

    if random:
        z = torch.randn(total_images, z_dim[0], z_dim[1], z_dim[2])
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

    generative_images, _ = model.decoder(z)
    # torchvision.utils.save_image(generative_images, 'GenerativePlot.png', nrow=image_per_row)
    generative_images = torchvision.utils.make_grid(generative_images, nrow=image_per_row).permute(1, 2,
                                                                                                   0).detach().cpu()
    plt.imshow(generative_images)
    plt.show()
    return generative_images


def ScatterPlot(elbo, label='', x_index=[], plot=None, marker='x', color='g', save_name=None, show=False):
    '''
    画出elbo1的散点图
    '''
    if len(x_index) == 0:
        x_index = torch.range(0, len(elbo))

    if not plot:
        plot = plt

    elbo = elbo.detach().cpu().numpy()
    plot.scatter(x=x_index, y=elbo, marker=marker, c=color, label=label)
    plot.legend()

    if save_name:
        plot.savefig(save_name)

    if show:
        plot.show()

    return plot


Markers = [
    '.',  # point marker
    'o',  # circle marker
    '*',  # star marker
    '+',  # plus marker
    'x',  # x marker
    'v',  # triangle_down marker
    '^',  # triangle_up marker
    '<',  # triangle_left marker
    '>',  # triangle_right marker
    '1',  # tri_down marker
    '2',  # tri_up marker
    '3',  # tri_left marker
    '4',  # tri_right marker
    's',  # square marker
    'p',  # pentagon marker
    'h',  # hexagon1 marker
    'H',  # hexagon2 marker
    ',',  # pixel marker
    'D',  # diamond marker
    'd',  # thin_diamond marker
    '|',  # vline marker
    '_',  # hline marker
]

Colors = [
    'b',  # blue
    'g',  # green
    'r',  # red
    'c',  # cyan
    'm',  # magenta
    'y',  # yellow
    'k',  # black
    'w',  # white
]


def ListScatterPlot(elbolist, elbolabel, marker=Markers, color=Colors, filename=None):
    '''
    画出序列散点图
    '''

    plot = None

    show = False

    save_name = None

    elbo_num = [elbo.size(0) for elbo in elbolist]
    elbo_num.insert(0, 0)
    elbo_num = torch.cumsum(torch.tensor(elbo_num), dim=0)
    index = torch.randperm(elbo_num[-1])

    for i in range(len(elbolist)):
        if i == len(elbolist) - 1:
            save_name = filename
            show = True
        plot = ScatterPlot(elbo=elbolist[i], label=elbolabel[i], x_index=index[elbo_num[i]:elbo_num[i + 1]], plot=plot,
                           marker=marker[i % len(marker)], color=color[i % len(color)], save_name=save_name, show=show)

    return plot
