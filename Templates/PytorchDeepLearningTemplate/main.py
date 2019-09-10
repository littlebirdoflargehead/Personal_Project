import torch
import models
from torch.utils.data import DataLoader
from torchnet import meter
from data import DogCat

from config import Config



def train(Config):
    '''
    模型训练的整个流程，包括：
    step1: 定义模型
    step2: 数据
    step3: 目标函数与优化器
    step4: 统计指标：平滑处理之后的损失，还有混淆矩阵
    训练并统计
    '''
    # step1: 定义模型
    model = getattr(models,Config.model)()
    if Config.load_model_path:
        model.load(Config.load_model_path)
    if Config.use_gpu:
        model.cuda()

    # step2: 数据
    train_dataset = DogCat(root=Config.train_data_root,train=True,test=False)
    val_dataset = DogCat(root=Config.train_data_root,train=False,test=False)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=Config.batch_size,shuffle=True,num_workers=Config.num_workers)
    val_dataloader = DataLoader(dataset=val_dataset,batch_size=Config.batch_size,shuffle=False,num_workers=Config.num_workers)

    # step3: 目标函数与优化器
    criterion = torch.nn.CrossEntropyLoss()
    lr = Config.lr
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=Config.weight_decay)

    # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # 训练
    for epoch in range(Config.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for i ,(inputs,targets) in enumerate(train_dataloader):
            if Config.use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,targets)

            loss.backward()
            optimizer.step()

            # 更新统计指标
            loss_meter.add(loss.item())
            confusion_matrix.add(outputs.data,targets.data)

            if i%Config.print_freq==Config.print_freq-1:
                acc = val(model,val_dataloader,Config.use_gpu)
                print('Epoch:',epoch+1,'Round:',i+1,'Loss:',loss_meter.value()[0],'Accuracy:',acc)

        model.save()

        # 计算验证集上的指标
        # val(model,val_dataloader,Config.use_gpu)

        # 当损失函数不再下降时，调整学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr*Config.lr_decay
            print('LearningRate:',lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


def val(model,dataloader,use_gpu):
    '''
    计算模型在验证集上的准确率等信息
    '''
    model.eval()

    confusion_matrix = meter.ConfusionMeter(2)
    sum_accurate = 0
    sum_total = 0
    for i ,(inputs,targets) in enumerate(dataloader):
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()
        outputs = model(inputs)

        accurate, total = accurate_count(outputs, targets)
        sum_accurate += accurate
        sum_total += total
        confusion_matrix.add(outputs.detach().squeeze(), targets.type(torch.LongTensor))

    model.train()

    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / \
               (cm_value.sum())
    accuracy = sum_accurate/sum_total
    return  accuracy


def test(Config):

    # 模型
    model = getattr(models,Config.model)().eval()
    if Config.load_model_path:
        model.load(Config.load_model_path)
    if Config.use_gpu: model.cuda()

    # 数据
    test_dataset = DogCat(root=Config.test_data_root,train=False,test=True)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=Config.batch_size,shuffle=False,num_workers=Config.num_workers)
    result = []
    for i, (data,path) in enumerate(test_dataset):
        inputs = data.to(Config.device)
        outputs = model(inputs)


def accurate_count(outputs,targets):

    predict = torch.argmax(outputs,dim=1)
    count = 0
    for i in range(outputs.shape[0]):
        if predict[i]==targets[i]:
            count +=1

    return count,outputs.shape[0]


train(Config)
# test(Config)