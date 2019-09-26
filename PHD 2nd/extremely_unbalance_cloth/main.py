import torch
import numpy as np
import models
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from data import Sub_MNIST, GoodOrBadCloth
from utils import VAE_Loss, ImageVsReImagePlot, GenerativePlot, ListScatterPlot
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
    # step1: 数据
    # 建立大数据集与小数据集，与分别的DataLoader
    Good_Dataset_train = GoodOrBadCloth(root=Config.train_data_root, good=True, train=True, validation=False)
    Bad_Dataset_train = GoodOrBadCloth(root=Config.train_data_root, good=False, train=True, validation=False)
    Good_DataLoader_train = DataLoader(dataset=Good_Dataset_train, batch_size=Config.batch_size, shuffle=True,
                                       num_workers=Config.num_workers)
    Bad_DataLoader_train = DataLoader(dataset=Bad_Dataset_train, batch_size=Config.batch_size, shuffle=True,
                                      num_workers=Config.num_workers)

    # step2: 定义模型
    model = getattr(models, Config.model)()
    for name, param in model.named_parameters():
        print(name, param.shape)

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
    #     for i, images in enumerate(Good_DataLoader_train):
    #         if Config.use_gpu:
    #             images = images.to(Config.device)
    #
    #         optimizer.zero_grad()
    #         re_img_mean, re_img_logvar, z_mean, z_logvar = model(images)
    #         loss = torch.sum(VAE_Loss(images, re_img_mean, re_img_logvar, z_mean, z_logvar))
    #         ave_loss[epoch] = ave_loss[epoch] + loss.item() / len(Good_Dataset_train)
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #         # if i%Config.print_freq==Config.print_freq-1:
    #         #     # 当达到指定频率时，显示损失函数并画图
    #         #     print('Epoch:',epoch+1,'Round:',i+1,'Loss:',loss.item())
    #         #     ImageVsReImagePlot(images,re_img_mean,Config)
    #
    #     print('Epoch:', epoch + 1, 'AverageLoss:', ave_loss[epoch])
    #
    #     if epoch % 10 == 0:
    #         ImageVsReImagePlot(images, re_img_mean, Config)
    #         GenerativePlot(model, Config, random=True)
    #
    #
    # save_path = model.save()
    # print(save_path)

    # 训练阶段二，增大模型在大数据集与小数据集上的似然函数的差距
    ave_loss_g = np.zeros(Config.max_epoch)
    ave_loss_b = np.zeros(Config.max_epoch)
    ave_loss_dif = np.zeros(Config.max_epoch)
    std_loss_g = np.zeros(Config.max_epoch)
    std_loss_b = np.zeros(Config.max_epoch)
    Config._parse({'load_model_path': None})
    for epoch in range(Config.max_epoch):
        for i, images_g in enumerate(Good_DataLoader_train):
            for images_b in iter(Bad_DataLoader_train):
                # 当前方法是使用两个循环来读取两个不同迭代器中的数据
                break
            if Config.use_gpu:
                images_g = images_g.to(Config.device)
                images_b = images_b.to(Config.device)

            optimizer.zero_grad()
            re_img_mean_g, re_img_logvar_g, z_mean_g, z_logvar_g = model(images_g)
            loss_g = VAE_Loss(images_g, re_img_mean_g, re_img_logvar_g, z_mean_g, z_logvar_g)

            re_images_b, re_img_logvar_b, z_mean_b, z_logvar_b = model(images_b)
            loss_b = VAE_Loss(images_b, re_images_b, re_img_logvar_b, z_mean_b, z_logvar_b)

            loss_dif = torch.mean(loss_g) - torch.mean(loss_b)

            # 统计每一个epoch中，两组数据的loss的平均值及loss的标准差值
            ave_loss_g[epoch] = ave_loss_g[epoch] + torch.sum(loss_g).item() / len(Good_Dataset_train)
            std_loss_g[epoch] = (i * std_loss_g[epoch] + torch.std(loss_g).item()) / (i + 1)
            ave_loss_b[epoch] = (i * ave_loss_b[epoch] + torch.mean(loss_b).item()) / (i + 1)
            std_loss_b[epoch] = (i * std_loss_b[epoch] + torch.std(loss_b).item()) / (i + 1)
            ave_loss_dif[epoch] = ave_loss_dif[epoch] + loss_dif.item() * loss_g.shape[0] / len(Good_Dataset_train)

            Loss = loss_dif + 10 * torch.pow(torch.mean(torch.clamp_min(loss_g - 21296.7370, 0)), 2) + torch.var(
                loss_b) + torch.var(loss_g)
            # + 10 * torch.pow(torch.mean(loss_m) - 105.87, 2) + torch.var(loss_f) + torch.var(loss_m)
            # Loss = -torch.pow(loss_dif,2)/(10*torch.var(loss_f)+torch.var(loss_m))+10*torch.pow(torch.mean(loss_m)-105.87,2)

            Loss.backward()
            optimizer.step()

        print('Epoch:', epoch + 1, 'AverageLossDifference:', ave_loss_dif[epoch],
              'AverageLoss_GoodDataSet:', ave_loss_g[epoch], 'AverageLoss_BadDataSet:', ave_loss_b[epoch])

        # threshold = (ave_loss_f-ave_loss_m)*std_loss_m/(std_loss_m+std_loss_f)+ave_loss_m
        # print('阈值为：',threshold[epoch])

        threshold = [23000 + s * 2000 for s in range(10)]
        accuracy_list, confusion_matrix_list, ELBO_LIST = test(Config, model, ValidOrTest='valid',
                                                               threshold=threshold, epoch=epoch)

        if epoch % 8 == 0:
            GenerativePlot(model, Config, random=True)


    save_path = model.save()
    print(save_path)
    return save_path, ave_loss_g, ave_loss_b


def test(Config, model=None, ValidOrTest='test', threshold=[150], epoch=0):
    '''
    模型训练的整个流程，包括：
    step1: 数据
    step2: 读取模型
    step3: 统计指标：平滑处理之后的损失，还有混淆矩阵（无监督训练时不需要）
    '''

    # step1: 数据
    if ValidOrTest == 'test':
        train = False
        validation = True
    elif ValidOrTest == 'valid':
        train = True
        validation = True
    elif ValidOrTest == 'train':
        train = True
        validation = False
    else:
        Warning('非法输入!!!')
    Good_Dataset_test = GoodOrBadCloth(root=Config.train_data_root, good=True, train=train, validation=validation)
    Bad_Dataset_test = GoodOrBadCloth(root=Config.train_data_root, good=False, train=train, validation=validation)
    Good_DataLoader_test = DataLoader(dataset=Good_Dataset_test, batch_size=Config.batch_size // 2, shuffle=True,
                                      num_workers=Config.num_workers)
    Bad_DataLoader_test = DataLoader(dataset=Bad_Dataset_test, batch_size=Config.batch_size // 2, shuffle=True,
                                     num_workers=Config.num_workers)

    # step2: 读取模型
    if not model:
        model = getattr(models, Config.model)()
    if Config.load_model_path:
        model.load(Config.load_model_path)
    if Config.use_gpu:
        model.to(Config.device)

    # step3：统计指标
    ELBO_LIST = []
    DataLoaders = [Good_DataLoader_test, Bad_DataLoader_test]
    for dataset_idx in range(len(DataLoaders)):

        ELBO = torch.tensor([]).to(Config.device)

        for i, images in enumerate(DataLoaders[dataset_idx]):
            if Config.use_gpu:
                images = images.to(Config.device)

            # 对每个样本计算平均的evidence lower bound
            z_mean, z_logvar = model.encoder(images)
            n = 10
            elbo = 0
            for k in range(n):
                z, _ = model.reparameter_trick(z_mean, z_logvar)
                re_images_mean, re_img_logvar = model.decoder(z)
                loss = VAE_Loss(images, re_images_mean, re_img_logvar, z_mean, z_logvar).detach()
                elbo = elbo + loss / n

            # 统计每一类的ELBO
            ELBO = torch.cat([ELBO, elbo])

        ELBO_LIST.append(ELBO)

    # 对不同数据集中的ELBO值进行画图
    if epoch % 8 == 0:
        ListScatterPlot(ELBO_LIST, marker=['o'], filename='elbo{}.png'.format(epoch))

    # 统计样本是否被正确分类
    accuracy_list = []
    confusion_matrix_list = []
    for j in range(len(threshold)):
        confusion_matrix = np.zeros([len(DataLoaders), len(DataLoaders)])

        for dataset_idx in range(len(DataLoaders)):
            confusion_matrix[dataset_idx, 0] += torch.sum(ELBO_LIST[dataset_idx] < threshold[j])
            confusion_matrix[dataset_idx, 1] += torch.sum(ELBO_LIST[dataset_idx] >= threshold[j])

        confusion_matrix_list.append(confusion_matrix)
        accuracy_list.append((confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix))
        print('阈值为:', threshold[j], '正确率为：', accuracy_list[j] * 100, '%')
        print(confusion_matrix)

    rate_bad = torch.sum(ELBO_LIST[1]<ELBO_LIST[0].max()).float()/ELBO_LIST[1].size(0)
    rate_good = torch.sum(ELBO_LIST[0]>ELBO_LIST[1].min()).float()/ELBO_LIST[0].size(0)
    if rate_bad==0 and rate_good<0.01:
        print('满足要求！！')

    print('坏布ELBO浸入率为',rate_bad.item()*100,'%','好布ELBO浸入率为',rate_good.item()*100,'%')

    return accuracy_list, confusion_matrix_list, ELBO_LIST


# Config._parse({'load_model_path': 'checkpoints/vae-190919_15:15:19.pth'})  # tensor z / with sigmoid / determined logvar
# Config._parse({'load_model_path': 'checkpoints/vae-190919_22:20:59.pth'})  # 训练阶段2 # tensor z / with sigmoid / determined logvar
# Config._parse({'load_model_path': 'checkpoints/vae-190923_14:54:42.pth'})  # 训练阶段2 # tensor z / with sigmoid / determined logvar
# Config._parse({'load_model_path': 'checkpoints/vae-190925_13:26:46.pth'})  # 训练阶段1(validation) # tensor z / with sigmoid / determined logvar
Config._parse({'load_model_path': 'checkpoints/vae-190926_09:16:43.pth'})  # 训练阶段2(validation) # tensor z / with sigmoid / determined logvar
save_path, ave_loss_m, ave_loss_f = train(Config)

# Config._parse({'load_model_path':'checkpoints/vae-190912_14:48:28.pth'})
# test(Config,threshold=155)
