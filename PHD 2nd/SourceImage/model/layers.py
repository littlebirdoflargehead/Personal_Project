import math
from math import ceil, sqrt
import numpy as np

import torch

import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


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
        output = torch.matmul(adj.to_dense(), support)
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

        nnz = adj._nnz()
        self.weight = Parameter(torch.zeros(nnz))
        self.weight.data.normal_(std=1. / torch.sparse.sum(adj, 0).to_dense().mean().sqrt())

        self.indices = adj.indices()
        self.size = adj.size()

        self.special_spmm = SpecialSpmm()

    def forward(self, x):
        s = self.special_spmm(self.indices, self.weight.exp(), self.size,
                              torch.ones(x.shape[0], x.shape[1], 1).to(x.device))
        x = self.special_spmm(self.indices, self.weight.exp(), self.size, x)
        x = x.div(s + 1e-20)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.size[1]) + ' -> ' \
               + str(self.size[0]) + ')'


class GraphAttentionLayer(Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, nheads, adj):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.nheads = nheads
        self.adj = adj

        std = 1 / sqrt(out_features)
        self.W = Parameter(torch.zeros(size=(nheads, in_features, out_features)))
        self.W.data.uniform_(-std, std)
        self.a = Parameter(torch.zeros(size=(nheads, out_features, 2)))
        self.a.data.uniform_(-std / sqrt(adj.shape[0]), std / sqrt(adj.shape[0]))

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.batch_spmm = BatchSpmm()
        self.batch_adj = dict()

    def forward(self, input):
        device = input.device
        edge = self.adj.indices()

        h_prime_list = []
        for head in range(self.nheads):
            h = torch.matmul(input, self.W[head])
            batch_size, N, out_features = h.size()

            edge_e = h.matmul(self.a[head])
            edge_e = edge_e[:, edge[0], 0] + edge_e[:, edge[1], 1]
            edge_e = torch.exp(-self.leakyrelu(edge_e.squeeze()).clamp(-50, 50))

            if str(batch_size) in self.batch_adj.keys():
                batch_adj = self.batch_adj[str(batch_size)]
            else:
                indices = edge.clone()
                indices[1] = torch.arange(indices.shape[1], device=device)
                shape = torch.tensor([N, indices.shape[1]], device=device)
                indices_batch = torch.cat([indices + i * shape.view(-1, 1) for i in range(batch_size)], 1)
                batch_adj = torch.sparse_coo_tensor(indices_batch, torch.ones_like(edge_e.view(-1)),
                                                    torch.Size(shape * batch_size))
                self.batch_adj[str(batch_size)] = batch_adj

            e_rowsum = batch_adj.matmul(edge_e.view(-1, 1)).view(batch_size, -1, 1)
            # e_rowsum = self.batch_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(batch_size, N, 1), device=device))
            edge_alpha = edge_e.div(e_rowsum[:, edge[0], :].squeeze() + 1e-20)
            h_prime = batch_adj.matmul(edge_alpha.view(-1, 1) * h[:, edge[1]].view(-1, h.shape[-1])).view(batch_size,
                                                                                                          -1,
                                                                                                          h.shape[-1])
            # h_prime = self.batch_spmm(edge, edge_alpha, torch.Size([N, N]), h)
            if torch.isnan(h_prime).any():
                h_prime[torch.isnan(h_prime)] = 0.

            if self.nheads > 1:
                h_prime_list.append(F.elu(h_prime))
            else:
                return h_prime

        return torch.cat(h_prime_list, dim=-1)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + ' -> ' + str(
            self.out_features * self.nheads) + ')'


class ClusterAttention(Module):

    def __init__(self, clusters, adj, out_features):
        super(ClusterAttention, self).__init__()
        self.clusters = clusters
        self.adj_clusters = adj
        self.cls = torch.sparse.sum(adj, dim=1).to_dense().unsqueeze(-1)
        self.in_features = len(clusters)
        self.out_features = out_features

        std = 1 / sqrt(self.out_features)
        self.att = Parameter(torch.full(size=(self.in_features, self.out_features), fill_value=-std))
        self.att.data.uniform_(-std, std)
        # self.att.data.normal_(std=std).clamp_(-3*std, 3*std)

    def forward(self, x):
        s = (self.adj_clusters.to_dense() / self.cls).matmul(x.pow(2))
        s_max = s.max(dim=-1, keepdim=True)[0]
        att_clusters = F.softmax(-F.leaky_relu(s / s_max * self.att.expand_as(s), 0.2).sum(-1, keepdim=True), dim=1)
        att_vertices = self.adj_clusters.t().to_dense().matmul(att_clusters)
        att_vertices = att_vertices / att_vertices.max(dim=1, keepdim=True)[0]
        return att_vertices, att_clusters


class ClusterAttention2(Module):

    def __init__(self, clusters, adj, in_features, att_features):
        super(ClusterAttention2, self).__init__()
        self.clusters = clusters
        self.adj_clusters = adj
        self.cls = torch.sparse.sum(adj, dim=1).to_dense().unsqueeze(-1)
        self.in_features = in_features
        self.att_features = att_features

        std = 1 / sqrt(self.att_features)
        self.att = Parameter(torch.full(size=(len(clusters), att_features), fill_value=-std))
        self.att.data.uniform_(-std, std)
        # self.att.data.normal_(std=std).clamp_(-3*std, 3*std)

        std = 1 / sqrt(self.in_features)
        self.key = Parameter(torch.full(size=(in_features, att_features), fill_value=-std))
        self.key.data.normal_(std=std)

    def forward(self, x):
        s = (self.adj_clusters.to_dense() / self.cls).matmul(x.matmul(self.key).pow(2))
        s_max = s.max(dim=-1, keepdim=True)[0]
        att_clusters = F.softmax(-F.leaky_relu(s / s_max * self.att.expand_as(s), 0.2).sum(-1, keepdim=True), dim=1)
        att_vertices = self.adj_clusters.t().to_dense().matmul(att_clusters)
        att_vertices = att_vertices / att_vertices.max(dim=1, keepdim=True)[0]
        return att_vertices, att_clusters


class VerticesAttention(Module):

    def __init__(self, adj, in_features, att_features, att=None):
        super(VerticesAttention, self).__init__()
        self.adj = adj
        self.in_features = in_features
        self.att_features = att_features

        if att is None:
            std = 1 / sqrt(self.att_features)
            self.att = Parameter(torch.full(size=(adj.shape[0], att_features), fill_value=-std))
            self.att.data.uniform_(-std, std)
            # self.att.data.normal_(std=std).clamp_(-3*std, 3*std)
        else:
            self.att = att

        std = 1 / sqrt(self.in_features)
        self.key = Parameter(torch.full(size=(in_features, att_features), fill_value=-std))
        self.key.data.normal_(std=std)

    def forward(self, x):
        s = torch.matmul(x, self.key).pow(2)
        s_max = s.max(dim=-1, keepdim=True)[0]+1e-20
        att_vertices = F.softmax(-F.leaky_relu(s / s_max * self.att.expand_as(s), 0.2).sum(-1, keepdim=True), dim=1)
        att_vertices = att_vertices / att_vertices.max(dim=1, keepdim=True)[0]
        return att_vertices


class BatchSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert not indices.requires_grad
        batch_size = values.shape[0]
        ctx.batch_size = batch_size
        shape = torch.tensor([shape[0], shape[1]], device=b.device)
        indices_batch = torch.cat([indices + i * shape.view(-1, 1) for i in range(batch_size)], 1)
        a = torch.sparse_coo_tensor(indices_batch, values.view(-1), torch.Size(shape * batch_size))
        ctx.save_for_backward(indices, a, b)
        return torch.spmm(a, b.view(-1, b.shape[-1])).view(batch_size, -1, b.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        indices, a, b = ctx.saved_tensors
        grad_values = grad_b = None
        batch_size = ctx.batch_size
        if ctx.needs_input_grad[1]:
            grad_values = torch.sum(grad_output[:, indices[0], :] * b[:, indices[1], :], dim=-1)
        if ctx.needs_input_grad[3]:
            grad_b = torch.spmm(a.t(), grad_output.view(-1, grad_output.shape[-1])).view(batch_size, -1,
                                                                                         grad_output.shape[-1])
        return None, grad_values, None, grad_b


class BatchSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return BatchSpmmFunction.apply(indices, values, shape, b)


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert not indices.requires_grad
        a = torch.sparse_coo_tensor(indices, values, shape).coalesce()
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a.to_dense(), b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        indices = a.indices()
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_values = torch.sum(grad_output[:, indices[0], :] * b[:, indices[1], :], dim=[0, -1])
        if ctx.needs_input_grad[3]:
            grad_b = a.to_dense().t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)
