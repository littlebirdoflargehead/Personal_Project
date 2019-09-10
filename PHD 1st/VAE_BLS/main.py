import torch
from models import Decoder, BLS
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from utils import VAE_Loss, ImageVsReImagePlot, GenerativePlot, pinv
from config import Config
import matplotlib.pyplot as plt
import seaborn as sns


torch.set_default_tensor_type(torch.DoubleTensor)

def train(Config):
    '''
    模型训练的整个流程，包括：
    step1: 数据
    step2: 定义模型
    step3: 目标函数与优化器
    step4: 统计指标：平滑处理之后的损失，还有混淆矩阵（无监督训练时不需要）
    训练并统计
    '''

    # step1: 数据
    train_dataset = torchvision.datasets.MNIST(root=Config.train_data_root, train=True, transform=transforms.ToTensor(),
                                               download=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=Config.batch_size, shuffle=True,
                                  num_workers=Config.num_workers)

    # step2: 定义模型
    decoder = Decoder(z_dim=10)
    if Config.load_model_path:
        decoder.load(Config.load_model_path)
    if Config.use_gpu:
        decoder.to(Config.device)

    # step3: 目标函数与优化器
    lr = Config.lr
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=Config.weight_decay)

    # 训练
    N1 = Config.N1
    N2 = Config.N2
    N3 = Config.N3

    bls = BLS(N1,N2,N3,28*28,10)

    images,_ = iter(train_dataloader).next()
    if Config.use_gpu:
        images = images.cuda()
    images = images.double().view(-1, 28 * 28)
    FeatureMaps, EnhancementNodes = bls(images.cpu(), train=True)
    # 使用矩阵伪逆求出对应的参数矩阵W
    A = torch.cat([FeatureMaps, EnhancementNodes], dim=1)
    A_inv = pinv(A)
    W = 0.1 * torch.randn(A.size()[1], 2 * decoder.z_dim)

    for epoch in range(Config.max_epoch):
        parameters = A.mm(W)

        # 使用reparameter trick
        # 注意：由于pytorch中并不会保存中间变量的梯度值(故须要使用hook机制进行保存)
        mu = parameters[:, :decoder.fc1.in_features].to(Config.device)
        mu.register_hook(save_grad('mu'))
        logvar = parameters[:, decoder.fc1.in_features:].to(Config.device)
        logvar.register_hook(save_grad('logvar'))

        std = torch.exp(0.5 * logvar)
        epslon = torch.randn_like(mu)
        z = epslon * std + mu

        # 将encoder产生的z放入到encoder中，并计算loss
        optimizer.zero_grad()
        re_images = decoder(z)
        loss = VAE_Loss(images, re_images, mu, logvar)

        loss.backward()
        print(grads['mu'], grads['logvar'])

        mu = mu - 100 * grads['mu']
        logvar = logvar - 100 * grads['logvar']

        W = W - A_inv.mm(torch.cat(100*[grads['mu'],grads['logvar']],dim=1).cpu())
        WW = A_inv.mm(torch.cat(100*[mu,logvar],dim=1))

        optimizer.step()

    for epoch in range(Config.max_epoch):

        for i, (images,_) in enumerate(train_dataloader):
            if Config.use_gpu:
                images = images.cuda()
            images = images.double().view(-1, 28 * 28)

            FeatureMaps, EnhancementNodes = bls(images.cpu(),train=True)

            _,enhancementnode = bls.AddEnhancementNodes(50)


            # 使用矩阵伪逆求出对应的参数矩阵W
            A = torch.cat([FeatureMaps, EnhancementNodes], dim=1)
            A_inv = pinv(A)

            W = 0.1*torch.randn(A.size()[1], 2 * decoder.z_dim)
            parameters = A.mm(W)

            # 使用reparameter trick
            # 注意：由于pytorch中并不会保存中间变量的梯度值(故须要使用hook机制进行保存)
            mu = parameters[:, :decoder.fc1.in_features].to(Config.device)
            mu.register_hook(save_grad('mu'))
            logvar = parameters[:, decoder.fc1.in_features:].to(Config.device)
            logvar.register_hook(save_grad('logvar'))

            # sns.set_style('darkgrid')
            # sns.distplot(mu.view(-1).cpu().detach())
            # plt.show()
            # sns.set_style('darkgrid')
            # sns.distplot(logvar.view(-1).cpu().detach())
            # plt.show()


            std = torch.exp(0.5 * logvar)
            epslon = torch.randn_like(mu)
            z = epslon*std+mu

            # 将encoder产生的z放入到encoder中，并计算loss
            optimizer.zero_grad()
            re_images = decoder(z)
            loss = VAE_Loss(images,re_images,mu,logvar)

            loss.backward()
            print(grads['mu'],grads['logvar'])

            mu = mu - lr*grads['mu']
            logvar = logvar - lr*grads[logvar]

            optimizer.step()



            if i % Config.print_freq == Config.print_freq - 1:
                # 当达到指定频率时，显示损失函数并画图
                print('Epoch:', epoch + 1, 'Round:', i + 1, 'Loss:', loss.item())
                ImageVsReImagePlot(images, re_images, Config)

        decoder.save()
        GenerativePlot(decoder, Config)


def Marginal_Likelihood_Evaluate(model, Config):
    # step1: 数据
    test_dataset = torchvision.datasets.MNIST(root=Config.test_data_root, train=False, transform=transforms.ToTensor(),
                                              download=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False, num_workers=Config.num_workers)

    # step: 抽样计算边缘
    L = 5
    LikeLihood = 0
    for data, _ in iter(test_dataloader):
        if Config.use_gpu:
            images = data.cuda()
        images = images.view(-1, 28 * 28)
        mu, logvar = model.encoder(images)
        loss = 0
        for l in range(L):
            z, epslon = model.reparameter_trick(mu, logvar)
            re_images = model.decoder(z)

            BCE = torch.sum(torch.log(re_images) * images + torch.log(1 - re_images) * (1 - images), dim=1)
            KLD = 0.5 * torch.sum(torch.pow(epslon, 2) + logvar - torch.pow(z, 2), dim=1)
            loss = loss + torch.exp(BCE + KLD)
        loss = torch.sum(torch.log(loss / L))
        LikeLihood = LikeLihood + loss
    return LikeLihood


grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


train(Config)
