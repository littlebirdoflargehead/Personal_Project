import scipy.sparse as sp

import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConvolution, GraphConvolutionDense, ParameterizedAdj, ParameterizedAdjDense
from .BasicModule import BasicModule
from Cortex import cortex_loader, gain_loader
from utils import sparse_mx_to_torch_sparse_tensor, gain_to_sparse, normalize


class GCN(nn.Module):
    def __init__(self, k=5, dropout=0.5, Layers=[10], Cortex=None, Gain=None):
        super(GCN, self).__init__()

        if Cortex is None:
            Cortex = cortex_loader()
        if Gain is None:
            Gain = gain_loader()
        self.adj_vert = sparse_mx_to_torch_sparse_tensor(Cortex['VertConn'], normal=True)
        self.adj_sensor = gain_to_sparse(Gain, normal=True, percentage=99).t()

        self.Layers = Layers

        self.gc_senor = GraphConvolution(k, Layers[0])
        for i in range(len(Layers)):
            if i == len(Layers) - 1:
                setattr(self, 'gc{}'.format(i), GraphConvolution(Layers[i], k))
            else:
                setattr(self, 'gc{}'.format(i), GraphConvolution(Layers[i], Layers[i + 1]))

        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.gc_senor(x, self.adj_sensor))
        for i in range(len(self.Layers)):
            if i == len(self.Layers) - 1:
                x = getattr(self, 'gc{}'.format(i))(x, self.adj_vert)
            else:
                x = getattr(self, 'gc{}'.format(i))(x, self.adj_vert)
                x = F.relu(x)
                # x = F.dropout(x, self.dropout, training=self.training)
        return x


class GCN_d(BasicModule):
    def __init__(self, k=5, dropout=0.5, Layers=[10], Cortex=None, Gain=None, device='cpu'):
        super(GCN_d, self).__init__()

        self.model_name = 'gcn'

        if Cortex is None:
            Cortex = cortex_loader()
        if Gain is None:
            Gain = gain_loader()
        self.adj_vert = sparse_mx_to_torch_sparse_tensor(Cortex['VertConn'], normal=True).to_dense().to(device)
        self.adj_sensor = gain_to_sparse(Gain, normal=True, percentage=99).t().to_dense().to(device)

        self.Layers = Layers

        self.gc_senor = GraphConvolutionDense(k, Layers[0])
        for i in range(len(Layers)):
            if i == len(Layers) - 1:
                setattr(self, 'gc{}'.format(i), GraphConvolutionDense(Layers[i], k))
            else:
                setattr(self, 'gc{}'.format(i), GraphConvolutionDense(Layers[i], Layers[i + 1]))

        self.dropout = dropout

        self.to(device)

    def forward(self, x):
        x = F.relu(self.gc_senor(x, self.adj_sensor))
        for i in range(len(self.Layers)):
            if i == len(self.Layers) - 1:
                x = getattr(self, 'gc{}'.format(i))(x, self.adj_vert)
            else:
                x = getattr(self, 'gc{}'.format(i))(x, self.adj_vert)
                x = F.relu(x)
                # x = F.dropout(x, self.dropout, training=self.training)
        return x


class GCN_adj(BasicModule):
    def __init__(self, dropout=0.5, Layers=2, Cortex=None, Gain=None, device='cpu'):
        super(GCN_adj, self).__init__()

        self.model_name = 'gcn_adj'
        if Cortex is None:
            Cortex = cortex_loader()
        if Gain is None:
            Gain = gain_loader()
        VertConn = Cortex['VertConn']

        self.Layers = Layers

        self.adj_sensor = ParameterizedAdj(gain_to_sparse(Gain, normal=False, percentage=95, istorch=False).T)
        for i in range(Layers):
            setattr(self, 'adj_vert{}'.format(i + 1), ParameterizedAdj(VertConn + sp.eye(VertConn.shape[0])))

        self.to(device)

    def forward(self, x):
        x = self.adj_sensor(x)
        for i in range(self.Layers):
            x = getattr(self, 'adj_vert{}'.format(i + 1))(x)
            if i < self.Layers - 1:
                x = F.relu(x, inplace=True)
        return x


class GCN_ADJ(BasicModule):
    '''
    Graph Convolutional Network with Parameterized Adjacency Matrix Layer
    '''
    def __init__(self, k=5, dropout=0.5, Layers=[10], Cortex=None, Gain=None, device='cpu', perlayer=False):
        super(GCN_ADJ, self).__init__()

        self.model_name = 'gcn_adj'

        if Cortex is None:
            Cortex = cortex_loader()
        if Gain is None:
            Gain = gain_loader()
        VertConn = Cortex['VertConn']

        self.adj_vert = sparse_mx_to_torch_sparse_tensor(VertConn + sp.eye(VertConn.shape[0]),
                                                         normal=False).coalesce().to(device)
        self.adj_sensor = gain_to_sparse(Gain, normal=False, percentage=95).t().coalesce().to(device)

        self.Layers = Layers
        self.perlayer = perlayer

        self.fc_senor = nn.Linear(k, Layers[0])
        self.parm_adj_sensor = ParameterizedAdjDense(self.adj_sensor)
        for i in range(len(Layers)):
            if i == len(Layers) - 1:
                setattr(self, 'fc{}'.format(i), nn.Linear(Layers[i], k))
            else:
                setattr(self, 'fc{}'.format(i), nn.Linear(Layers[i], Layers[i + 1]))
            if perlayer:
                setattr(self, 'param_adj{}'.format(i), ParameterizedAdjDense(self.adj_vert))
            elif i == 0:
                self.param_adj = ParameterizedAdjDense(self.adj_vert)

        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc_senor(self.parm_adj_sensor(x, self.adj_sensor)))
        for i in range(len(self.Layers)):
            if self.perlayer:
                x = getattr(self, 'param_adj{}'.format(i))(x, self.adj_vert)
            else:
                x = self.param_adj(x, self.adj_vert)
            x = getattr(self, 'fc{}'.format(i))(x)
            if i < len(self.Layers) - 1:
                x = F.relu(x, inplace=True)
                # x = F.dropout(x, self.dropout, training=self.training)
        return x
