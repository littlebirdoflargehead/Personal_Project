import math
import numpy as np

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from utils import sparse_mx_to_torch_sparse_tensor


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolutionDense(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class ParameterizedAdj(Module):
    '''
    Parameterized Adjacency Matrix Layer in sparse format
    '''

    def __init__(self, adj):
        super(ParameterizedAdj, self).__init__()

        adj = adj.tocsc()
        value = torch.FloatTensor(adj.nnz)
        value.data.normal_(std=1./math.sqrt(adj.sum(0).mean()))

        adj = adj.tocoo()
        indices = torch.from_numpy(
            np.vstack((adj.row, adj.col)).astype(np.int64))
        shape = torch.Size(adj.shape)

        self.adj = Parameter(torch.sparse_coo_tensor(indices, value, shape))

    def forward(self, x):
        x = torch.matmul(self.adj.to_dense(), x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.adj.shape[1]) + ' -> ' \
               + str(self.adj.shape[0]) + ')'


class ParameterizedAdjDense(Module):
    '''
    Parameterized Adjacency Matrix Layer in dense format
    '''

    def __init__(self, adj):
        super(ParameterizedAdjDense, self).__init__()

        self.weight = Parameter(torch.FloatTensor(torch.Size(adj.shape)))
        self.weight.data.normal_(std=1./math.sqrt(adj.to_dense().sum(0).mean()))

    def forward(self, x, adj):
        x = torch.matmul(self.weight.sparse_mask(adj).to_dense(), x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.weight.shape[1]) + ' -> ' \
               + str(self.weight.shape[0]) + ')'
