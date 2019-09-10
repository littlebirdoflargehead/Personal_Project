import torch
import torch.nn as nn
import numpy as np
from sklearn import preprocessing
from scipy.sparse import diags
import time


def data_reshape(dataset):
    '''
    将数据由多维度数组转化为一维向量
    :param dataset: 多维度数组数据
    :return: 一维向量数据
    '''
    if isinstance(dataset,np.ndarray):
        dataset = torch.from_numpy(dataset)
    # 使用reshape而不使用view来对tensor进行变形，由于reshape能对non-contiguous tensor进行操作，而view则不能
    # 注意：无须除以255，由于数据类型为unit8除以255后只剩下0或1
    return torch.reshape(dataset,(dataset.shape[0],-1))


def one_hot(label):
    '''
    将torch的标签转化为one hot的形式
    :param label: torch的标签
    :return: numpy的one hot形式
    '''
    if ~isinstance(label,torch.Tensor):
        label = torch.from_numpy(np.array(label))
    label = label.view(-1, 1)
    label = torch.zeros(label.size()[0], 10).scatter_(1, label, 1).double()
    return label


def pinv(A, reg=2**-30):
    '''
    计算矩阵伪逆
    :param A: 须要计算伪逆的矩阵
    :param reg: 正则项系数
    :return: 矩阵伪逆
    '''
    # A_inv = torch.inverse(torch.t(A).mm(A) + reg * torch.eye(A.shape[1])).mm(torch.t(A))

    A = A.numpy()
    Q,R = np.linalg.qr(A)
    A_inv = np.linalg.solve(R,Q.T)
    return torch.from_numpy(A_inv)


def shrinkage(a, b):
    '''
    压缩参数
    :param a:
    :param b:
    :return:
    '''
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z


def sparse_bls(A, b):
    '''
    稀疏化参数
    :param A: 输出结果
    :param b: 输入
    :return: 经过稀疏化后的参数
    '''
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk


def iter_pinv(A,A_inv,H,c=2**-30):
    '''
    使用迭代公式计算矩阵的伪逆
    :param A: 原矩阵
    :param A_inv: 原矩阵伪逆
    :param H: 增加的矩阵块
    :return: 更新后的矩阵及其伪逆
    '''
    D = A_inv.mm(H)
    C = H - A.mm(D)
    if torch.allclose(C, torch.tensor([0]).double()):
        B = pinv(D, 1).mm(A_inv)
    else:
        B = pinv(C,c)
        s = B.mm(C)
    A_inv = torch.cat([A_inv-D.mm(B),B],dim=0)
    A = torch.cat([A,H],dim=1)
    ss = A_inv.mm(A)
    return A,A_inv



def ada_weight(A,train_y,SampleWeight,c=2**-30):
    '''
    输入输入矩阵，标签以及样本权重向量，得到对应的回归参数
    :param A: 输入矩阵A
    :param train_y: 训练集one-hot标签
    :param SampleWeight: 样本权重向量
    :param c:
    :return: tensor形式的回归权重W
    '''
    A = A.numpy()
    train_y = train_y.numpy()
    # 将一维向量SampleWeight转化为矩阵形式SampleWeightMatrix
    SampleWeightMatrix = diags(SampleWeight)
    s = SampleWeightMatrix.dot(A)
    s2 = SampleWeightMatrix.dot(train_y)
    S = A.T.dot(s)+c*np.eye(A.shape[1])
    S2 = A.T.dot(s2)
    W = np.mat(S).I.dot(S2)
    return torch.from_numpy(W)


def show_accuracy(predictLabel,Label):
    '''
    显示预测标签对比真实标签的正确率
    :param predictLabel: 预测的one-hot标签
    :param Label: 真实的one-hot标签
    :return: 正确率
    '''
    count = 0
    label = torch.argmax(Label,dim=1)
    predlabe = torch.argmax(predictLabel,dim=1)
    for j in range(Label.shape[0]):
        if label[j] == predlabe[j]:
            count += 1
    return count/Label.shape[0]


def show_accuracy_vote(train_out,Label):
    '''
    显示投票法集成学习中预测标签对比真实标签的正确率
    :param train_out: 预测标签数组
    :param Label: 真实的one-hot标签
    :return: 正确率
    '''
    count = 0
    label = torch.argmax(Label, dim=1)
    predlabe,_ = torch.mode(train_out,dim=1)
    for j in range(Label.shape[0]):
        if label[j] == predlabe[j]:
            count += 1
    return count/Label.shape[0]



def AveWeight(acc,method='exp'):
    '''
    当使用加权平均法进行集成时，输入准确率，返回对应的权重
    :param acc: 准确率
    :param method: 加权方式（可选择指数exp，对数log，等比例）
    :return: 对应的权重
    '''
    if method=='exp':
        w = np.exp(1/(1-acc))
    elif method=='log':
        w = -np.log(1-acc)
    elif method=='pro':
        w = acc
    else:
        w = np.ones(len(acc))
    return w/w.sum()



class FeatureMap(nn.Module):
    '''
    使用torch定义FeatureMap层，包括将权重稀疏化与前向传导
    '''
    def __init__(self, input_dim, output_dim):
        super(FeatureMap, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.set_init(self.fc)

    def set_init(self, layer):
        nn.init.uniform_(layer.weight, a=-1, b=1)
        nn.init.uniform_(layer.bias, a=-1, b=1)

    def forward(self, x, train=False):
        if train == True:
            # 在训练阶段，对权重使用重构稀疏化，并将稀疏化的权重保存在网络的参数中
            x_withbias = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1).numpy()
            Z = self.fc(x).detach().numpy()
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(Z)
            Z = scaler.transform(Z)
            sparse_w_b = sparse_bls(Z, x_withbias).T
            self.fc.state_dict()['weight'].copy_(torch.from_numpy(sparse_w_b[:x.shape[1]].T))
            self.fc.state_dict()['bias'].copy_(torch.from_numpy(sparse_w_b[x.shape[1]]).squeeze())
            # 对输出进行线性变换使其转化为[0,1]上的输出，保存极大极小值及其距离
            x = self.fc(x)
            self.min,_ = torch.min(x,dim=0)
            self.max,_ = torch.max(x,dim=0)
            self.dist = self.max-self.min
        else:
            # 若不在训练阶段，则直接输出结果
            x = self.fc(x)
        out = (x-self.min)/self.dist
        return out


class EnhancementNode(nn.Module):
    '''
    使用torch定义EnhancementNode层，包括初始化参数和前向传导
    '''
    def __init__(self, input_dim, output_dim):
        super(EnhancementNode, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.active = nn.Tanh()
        self.set_init(self.fc)

    def set_init(self, layer):
        nn.init.normal_(layer.weight, mean=0, std=0.05)
        nn.init.normal_(layer.bias, mean=0, std=0.005)

    def forward(self, x):
        x = self.fc(x)
        out = self.active(x)
        return out