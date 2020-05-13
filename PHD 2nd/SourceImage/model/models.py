import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConvolution,GraphConvolutionDense
from .BasicModule import BasicModule
from Cortex import cortex_loader, gain_loader
from utils import sparse_mx_to_torch_sparse_tensor, gain_to_sparse


class GCN(nn.Module):
    def __init__(self, k=5, dropout=0.5, Layers=[10], Cortex=None, Gain=None):
        super(GCN, self).__init__()

        if Cortex is None:
            Cortex = cortex_loader()
        if Gain is None:
            Gain = gain_loader()
        self.adj_vert = sparse_mx_to_torch_sparse_tensor(Cortex['VertConn'], normal=True)
        self.adj_sensor = gain_to_sparse(Gain, normal=True).t()

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
        self.adj_sensor = gain_to_sparse(Gain, normal=True).t().to_dense().to(device)

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
