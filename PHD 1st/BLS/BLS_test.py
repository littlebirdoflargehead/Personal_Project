import torch
from torch.utils.data import DataLoader
import torchvision
from ModelAndProcess import one_hot,data_reshape
from BLS_Versions import BLS,BLS_AddEnhanceNodes,BLS_Bagging,BLS_AdaBoost


# 程序初始化
torch.set_default_tensor_type(torch.DoubleTensor)

# 数据准备，使用torchvision中的数据集方便进行训练
data_dir = '/home2/liangjw/Documents/Pycharm_project/Pytorch_test/data'
train_dataset = torchvision.datasets.MNIST(root=data_dir,train=True,download=False,transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=False,num_workers=4)
test_dataset = torchvision.datasets.MNIST(root=data_dir,train=False,download=False,transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=False,num_workers=4)

# 将数据转化为一维向量，并且将标签转化为one-hot的形式/直接从dataset中读取数据而非从dataloader中读取数据会更快
traindata = data_reshape(train_dataset.data)
trainlabel = one_hot(train_dataset.targets)
testdata = data_reshape(test_dataset.data)
testlabel = one_hot(test_dataset.targets)

# traindata,trainlabel = iter(train_loader).next()
# traindata = traindata.view(-1,28*28)
# trainlabel = one_hot(trainlabel)
# testdata,testlabel = iter(test_loader).next()
# testdata = testdata.view(-1,28*28)
# testlabel = one_hot(testlabel)




# BLS系统
N1 = 10  #  feature nodes per window
N2 = 15  #  number of windows of feature nodes
N3 = 500 #  number of enhancement nodes
L = 20    #  # of incremental steps
M1 = 50  #  # of adding enhance nodes
s = 0.8  #  shrink coefficient
C = 2**-30 # Regularization coefficient



# print('-------------------BLS_BASE---------------------------')
# with torch.no_grad():
#     BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,12000)

print('-------------------BLS_ENHANCE------------------------')
with torch.no_grad():
    BLS_AddEnhanceNodes(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,L,M1)

print('-------------------BLS_Bagging------------------------')
N3 = 1200 #  number of enhancement nodes per bagging block
N = 10    # number of bagging block
with torch.no_grad():
    BLS_Bagging(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,N,Ensemble='ave',avemethod='exp')

print('-------------------BLS_AdaBoost------------------------')
N3 = 1200 #  number of enhancement nodes per bagging block
N = 10    # number of bagging block
with torch.no_grad():
    BLS_AdaBoost(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,N,Ensemble='ave',avemethod='log')