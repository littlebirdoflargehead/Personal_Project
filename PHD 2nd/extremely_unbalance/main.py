import torch
import numpy as np
import models
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from data import Sub_MNIST,GoodOrBadCloth
from utils import VAE_Loss, ImageVsReImagePlot, GenerativePlot
from config import Config


def train(Config):
    '''
    模型训练的整个流程，包括：
    step1: 数据
    step2: 定义模型
    step3: 目标函数与优化器
    step4: 统计指标：平滑处理之后的损失，还有混淆矩阵（无监督训练时不需要）
    训练并统计
    '''
    bad_Dataset = GoodOrBadCloth(root=Config.train_data_root,good=False,train=True)
    s = bad_Dataset[10]
    # step1: 数据
    # 建立大数据集与小数据集，与分别的DataLoader
    Much_Dataset = Sub_MNIST(root=Config.train_data_root, train=True, sublabel=Config.much_data_label)
    Few_Dataset = Sub_MNIST(root=Config.train_data_root, train=True, sublabel=Config.few_data_label, numperlabel=40)
    Much_DataLoader = DataLoader(dataset=Much_Dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    Few_DataLoader = DataLoader(dataset=Few_Dataset, batch_size=10, shuffle=True, num_workers=0)

    # step2: 定义模型
    model = getattr(models, Config.model)()
    if Config.load_model_path:
        model.load(Config.load_model_path)
    if Config.use_gpu:
        model.to(Config.device)

    # step3: 目标函数与优化器

    lr = Config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=Config.weight_decay)

    # 训练
    # 训练阶段一，将模型在大数据集上的似然函数提升(或是将Negative Lower Bound降低)，直到模型稳定
    # ave_loss = np.zeros(Config.max_epoch)
    # for epoch in range(Config.max_epoch):
    #
    #     for i ,(images,_) in enumerate(Much_DataLoader):
    #         if Config.use_gpu:
    #             images = images.cuda()
    #         images = images.view(-1, 28 * 28)
    #
    #         optimizer.zero_grad()
    #         re_images,mu,logvar = model(images)
    #         loss = torch.sum(VAE_Loss(images,re_images,mu,logvar))
    #         ave_loss[epoch] = ave_loss[epoch] + loss.item()/len(Much_Dataset)
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #
    #         # if i%Config.print_freq==Config.print_freq-1:
    #         #     # 当达到指定频率时，显示损失函数并画图
    #         #     print('Epoch:',epoch+1,'Round:',i+1,'Loss:',loss.item())
    #         #     ImageVsReImagePlot(images,re_images,Config)
    #
    #     GenerativePlot(model, Config,random=True)
    #     print('Epoch:',epoch+1,'AverageLoss:',ave_loss[epoch])
    #
    # save_path = model.save()
    # print(save_path)

    # 训练阶段二，增大模型在大数据集与小数据集上的似然函数的差距
    ave_loss_m = np.zeros(Config.max_epoch)
    ave_loss_f = np.zeros(Config.max_epoch)
    ave_loss_dif = np.zeros(Config.max_epoch)
    std_loss_m = np.zeros(Config.max_epoch)
    std_loss_f = np.zeros(Config.max_epoch)
    Config._parse({'load_model_path': None})
    for epoch in range(Config.max_epoch):
        for i, (images_m, _) in enumerate(Much_DataLoader):
            for (images_f, label_f) in iter(Few_DataLoader):
                # 当前方法是使用两个循环来读取两个不同迭代器中的数据
                break
            if Config.use_gpu:
                images_m = images_m.to(Config.device)
                images_f = images_f.to(Config.device)
            images_m = images_m.view(-1, 28 * 28)
            images_f = images_f.view(-1, 28 * 28)

            optimizer.zero_grad()
            re_images_m, mu_m, logvar_m = model(images_m)
            loss_m = VAE_Loss(images_m, re_images_m, mu_m, logvar_m)

            re_images_f, mu_f, logvar_f = model(images_f)
            loss_f = VAE_Loss(images_f, re_images_f, mu_f, logvar_f)

            loss_dif = torch.mean(loss_m) - torch.mean(loss_f)

            # 统计每一个epoch中，两组数据的loss的平均值及loss的标准差值
            ave_loss_m[epoch] = ave_loss_m[epoch] + torch.sum(loss_m).item() / len(Much_Dataset)
            std_loss_m[epoch] = (i * std_loss_m[epoch] + torch.std(loss_m).item()) / (i + 1)
            ave_loss_f[epoch] = (i * ave_loss_f[epoch] + torch.mean(loss_f).item()) / (i + 1)
            std_loss_f[epoch] = (i * std_loss_f[epoch] + torch.std(loss_f).item()) / (i + 1)
            ave_loss_dif[epoch] = ave_loss_dif[epoch] + loss_dif.item() * images_m.shape[0] / len(Much_Dataset)

            Loss = loss_dif + 10 * torch.pow(torch.mean(loss_m) - 64.5, 2) + torch.var(loss_f) + torch.var(loss_m)
            # + 10 * torch.pow(torch.mean(loss_m) - 105.87, 2) + torch.var(loss_f) + torch.var(loss_m)
            # Loss = -torch.pow(loss_dif,2)/(10*torch.var(loss_f)+torch.var(loss_m))+10*torch.pow(torch.mean(loss_m)-105.87,2)

            Loss.backward()
            optimizer.step()

        GenerativePlot(model, Config, random=True)
        print('Epoch:', epoch + 1, 'AverageLossDifference:', ave_loss_dif[epoch],
              'AverageLoss_MuchDataSet:', ave_loss_m[epoch], 'AverageLoss_FewDataSet:', ave_loss_f[epoch])

        # threshold = (ave_loss_f-ave_loss_m)*std_loss_m/(std_loss_m+std_loss_f)+ave_loss_m
        # print('阈值为：',threshold[epoch])

        for s in range(8):
            accuracy, confusion_matrix = test(Config, model, 90 + s * 10)
            print('阈值为:', 90 + s * 10, '正确率为：', accuracy * 100, '%')
            print(confusion_matrix)

    save_path = model.save()
    print(save_path)
    return save_path, ave_loss_m, ave_loss_f


def test(Config, model=None, threshold=150):
    '''
    模型训练的整个流程，包括：
    step1: 数据
    step2: 读取模型
    step3: 统计指标：平滑处理之后的损失，还有混淆矩阵（无监督训练时不需要）
    '''

    # step1: 数据
    Teat_Dataset = Sub_MNIST(root=Config.test_data_root, train=False,
                             sublabel=Config.few_data_label + Config.much_data_label)
    Test_DataLoader = DataLoader(dataset=Teat_Dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)

    # step2: 读取模型
    if not model:
        model = getattr(models, Config.model)()
    if Config.load_model_path:
        model.load(Config.load_model_path)
    if Config.use_gpu:
        model.to(Config.device)

    # step3：统计指标
    confusion_matrix = np.zeros([2, 2])
    for i, (images, labels) in enumerate(Test_DataLoader):
        if Config.use_gpu:
            images = images.to(Config.device)
        images = images.view(-1, 28 * 28)

        # 对每个样本计算平均的evidence lower bound
        mu, logvar = model.encoder(images)
        n = 5
        for k in range(n):
            z, _ = model.reparameter_trick(mu, logvar)
            re_images = model.decoder(z)
            loss = VAE_Loss(images, re_images, mu, logvar)
            if k == 0:
                elbo = loss / n
            else:
                elbo = elbo + loss / n

        # 统计样本是否被正确分类
        for j in range(len(labels)):
            real_index = 0 if labels[j] in Config.much_data_label else 1
            predict_index = 0 if elbo[j] < threshold else 1
            confusion_matrix[real_index, predict_index] += 1

    accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)
    return accuracy, confusion_matrix
    print(confusion_matrix)
    print('完成计算')
    print('正确率为：%f' % (accuracy * 100), '%')


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


# Config._parse({'load_model_path':'checkpoints/vae-190806_19:23:26.pth','max_epoch':80}) # 0 6 8/ 1 7
# Config._parse({'load_model_path':'checkpoints/vae-190812_08:59:25.pth','max_epoch':80}) # 2 3 5/ 1 7 9
Config._parse({'load_model_path': 'checkpoints/vae-190813_09:18:19.pth', 'max_epoch': 80})  # 1 7/ 0 6 8
save_path, ave_loss_m, ave_loss_f = train(Config)

# Config._parse({'load_model_path':'checkpoints/vae-190810_18:02:48.pth'})
# test(Config,threshold=155)
