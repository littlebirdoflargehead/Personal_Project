import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from .BasicModule import BasicModule
from utils import sparse_bls



class VAE(BasicModule):

    def __init__(self,image_size=28*28,hidden_dim=400,z_dim=20):
        super(VAE,self).__init__()

        self.model_name = 'vae'

        self.fc1 = nn.Linear(image_size,hidden_dim)
        self.fc21 = nn.Linear(hidden_dim,z_dim)
        self.fc22 = nn.Linear(hidden_dim,z_dim)
        self.fc3 = nn.Linear(z_dim,hidden_dim)
        self.fc4 = nn.Linear(hidden_dim,image_size)

    def encoder(self,x):
        h1 = self.fc1(x)
        mu = self.fc21(F.relu(h1))
        logvar = self.fc22(F.relu(h1))
        return mu,logvar

    def decoder(self,z):
        h2 = self.fc3(z)
        x = self.fc4(F.relu(h2))
        return torch.sigmoid(x)

    def reparameter_trick(self,mu,logvar):
        epslon = torch.randn_like(logvar)
        std = torch.exp(0.5*logvar)
        z = mu+epslon*std
        return z,epslon

    def forward(self, x):
        mu,logvar = self.encoder(x)
        z,_ = self.reparameter_trick(mu,logvar)
        out = self.decoder(z)
        return out,mu,logvar



class Decoder(BasicModule):
    """
    VAE中的decoder
    """
    def __init__(self,image_size=28*28,hidden_dim=400,z_dim=20):
        super(Decoder,self).__init__()

        self.model_name = 'decoder'
        self.z_dim = z_dim

        self.fc1 = nn.Linear(z_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,image_size)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        x = torch.sigmoid(self.fc2(h))
        return x



class FeatureMap(BasicModule):
    '''
    使用torch定义FeatureMap层，包括将权重稀疏化与前向传导
    '''
    def __init__(self, input_dim, output_dim):
        super(FeatureMap, self).__init__()

        self.model_name = 'featuremap'

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
            # 对输出进行线性变换使其转化为某个固定区间上的输出，保存极大极小值及其距离
            x = self.fc(x)
            self.min,_ = torch.min(x,dim=0)
            self.max,_ = torch.max(x,dim=0)
            self.dist = self.max-self.min
        else:
            # 若不在训练阶段，则直接输出结果
            x = self.fc(x)
        # 转化为区间[0,1]上的输出
        # out = (x-self.min)/self.dist
        # 转化为区间[-1,1]上的输出
        out = (2*x-self.min-self.max)/self.dist
        return out



class EnhancementNode(BasicModule):
    '''
    使用torch定义EnhancementNode层，包括初始化参数和前向传导
    '''
    def __init__(self, input_dim, output_dim):
        super(EnhancementNode, self).__init__()

        self.model_name = 'enhancement'

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



class BLS(BasicModule):
    '''
    将BLS中的FeatureLayers和EnhancementLayers的参数和方法都封装在该BLS中
    '''
    def __init__(self,N1,N2,N3,images_size,labels_size):
        super(BLS,self).__init__()

        self.model_name = 'BLS'

        self.FeatureLayers = []
        self.EnhancementLayers = []
        self.W = torch.randn(N1*N2+N3,labels_size)*0.1

        for _ in range(N2):
            featurelayer = FeatureMap(input_dim=images_size,output_dim=N1)
            self.FeatureLayers.append(featurelayer)

        enhancementlayer = EnhancementNode(input_dim=N1 * N2, output_dim=N3)
        self.EnhancementLayers.append(enhancementlayer)

    def forward(self, x,train=False):
        '''
        宽度网络的前向传播，输出FeatureMaps和EnhancementNodes对应的tensor
        :param x: 输入数据tensor
        :param train: FeatureMap是否train模式
        :return: FeatureMaps和EnhancementNodes对应的tensor
        '''
        for i in range(len(self.FeatureLayers)):
            featuremap = self.FeatureLayers[i](x,train=train)
            if i == 0:
                FeatureMaps = featuremap
            else:
                FeatureMaps = torch.cat([FeatureMaps, featuremap], dim=1)

        for i in range(len(self.EnhancementLayers)):
            enhancementnode = self.EnhancementLayers[i](FeatureMaps)
            if i == 0:
                EnhancementNodes = enhancementnode
            else:
                EnhancementNodes = torch.cat([EnhancementNodes,enhancementnode],dim=1)

        return FeatureMaps, EnhancementNodes

    def AddEnhancementNodes(self,N,FeatureMaps=None):
        '''
        Incremental Learning中加入EnhancementNodes，数量为N个，若不输入FeatureMaps，则仅返回对应的layer对象
        :param N: 须要增加的节点数
        :param FeatureMaps: FeatureMap层的输出
        :return: enhancementlayer和enhancementnode
        '''
        enhancementlayer = EnhancementNode(input_dim=self.FeatureLayers[0].fc.out_features,output_dim=N)
        self.EnhancementLayers.append(enhancementlayer)
        if FeatureMaps :
            enhancementnode = enhancementlayer(FeatureMaps)
        else:
            enhancementnode = None
        return enhancementlayer, enhancementnode