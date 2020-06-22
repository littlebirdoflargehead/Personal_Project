import scipy.sparse as sp
import numpy as np
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConvolution, GraphConvolutionDense, ParameterizedAdj, GraphAttentionLayer, ClusterAttention, \
    ClusterAttention2, VerticesAttention
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
        self.device = device

        self.fc_senor = nn.Linear(k, Layers[0])
        self.parm_adj_sensor = ParameterizedAdj(self.adj_sensor)
        for i in range(len(Layers)):
            if i == len(Layers) - 1:
                setattr(self, 'fc{}'.format(i), nn.Linear(Layers[i], k))
            else:
                setattr(self, 'fc{}'.format(i), nn.Linear(Layers[i], Layers[i + 1]))
            if perlayer:
                setattr(self, 'param_adj{}'.format(i), ParameterizedAdj(self.adj_vert))
            elif i == 0:
                self.param_adj = ParameterizedAdj(self.adj_vert)

        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc_senor(self.parm_adj_sensor(x)))
        for i in range(len(self.Layers)):
            if self.perlayer:
                x = getattr(self, 'param_adj{}'.format(i))(x)
            else:
                x = self.param_adj(x, self.adj_vert)
            x = getattr(self, 'fc{}'.format(i))(x)
            if i < len(self.Layers) - 1:
                x = F.relu(x, inplace=True)
                # x = F.dropout(x, self.dropout, training=self.training)
        return x


class Cluster_GCN_ADJ(BasicModule):
    '''
    Variational Graph Convolutional Network with Parameterized Adjacency Matrix Layer
    '''

    def __init__(self, k=5, dropout=0.5, Layers=[10], Cortex=None, Gain=None, device='cpu', perlayer=False):
        super(Cluster_GCN_ADJ, self).__init__()

        self.model_name = 'cgcn_adj'

        if Cortex is None:
            Cortex = cortex_loader()
        if Gain is None:
            Gain = gain_loader()

        if Cortex['Scales'].shape[0] >= len(Layers):
            raise ValueError('the number of hidden layers must be more than the number of scales')
        else:
            self.scales = np.vstack([Cortex['Scales'], np.arange(len(Layers) - Cortex['Scales'].shape[0], len(Layers))])

        cluster = []
        for i in range(len(Cortex['Clusters'])):
            scale = self.scales[0, i]
            cluster += Cortex['Clusters'][i]
            indices1 = np.empty([0], dtype=np.int)
            indices2 = np.empty([0], dtype=np.int)
            for j in range(len(Cortex['Clusters'][i])):
                indices1 = np.append(indices1, np.full_like(Cortex['Clusters'][i][j], j))
                indices2 = np.append(indices2, Cortex['Clusters'][i][j])
            indices = torch.from_numpy(np.vstack([indices1, indices2]))
            values = torch.ones(indices.shape[1])
            shape = torch.Size([len(Cortex['Clusters'][i]), Gain.shape[1]])
            adj_cluster = torch.sparse_coo_tensor(indices, values, shape).coalesce().to(device)
            setattr(self, 'adj_cluster{}'.format(scale), adj_cluster)
            setattr(self, 'param_adj_cluster{}'.format(scale), ParameterizedAdj(adj_cluster))
            layer = Layers[self.scales[1, i]]
            setattr(self, 'fc_cluster{}'.format(scale), nn.Linear(layer, 1))
        self.cluster = cluster

        indices1_all = np.empty([0], dtype=np.int)
        indices2_all = np.empty([0], dtype=np.int)
        for i in range(len(cluster)):
            indices1_all = np.append(indices1_all, np.full_like(cluster[i], i))
            indices2_all = np.append(indices2_all, cluster[i])
        indices_all = torch.from_numpy(np.vstack([indices1_all, indices2_all]))
        values_all = torch.ones(indices_all.shape[1])
        shape_all = torch.Size([len(cluster), Gain.shape[1]])
        self.adj_cluster_all = torch.sparse_coo_tensor(indices_all, values_all, shape_all).coalesce().to(device)

        VertConn = Cortex['VertConn']
        self.adj_vert = sparse_mx_to_torch_sparse_tensor(VertConn + sp.eye(VertConn.shape[0]),
                                                         normal=False).coalesce().to(device)
        self.adj_sensor = gain_to_sparse(Gain, normal=False, percentage=95).t().coalesce().to(device)

        self.Layers = Layers
        self.perlayer = perlayer
        self.device = device

        self.fc_senor = nn.Linear(k, Layers[0])
        self.parm_adj_sensor = ParameterizedAdj(self.adj_sensor)

        for i in range(len(Layers) - 1):
            setattr(self, 'fc{}'.format(i + 1), nn.Linear(Layers[i], Layers[i + 1]))
            if perlayer:
                setattr(self, 'param_adj{}'.format(i + 1), ParameterizedAdj(self.adj_vert))
            elif i == 0:
                self.param_adj = ParameterizedAdj(self.adj_vert)

        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc_senor(self.parm_adj_sensor(x)))
        Logit = []
        for i in range(len(self.Layers) - 1):
            i += 1
            if self.perlayer:
                x = getattr(self, 'param_adj{}'.format(i))(x)
            else:
                x = self.param_adj(x, self.adj_vert)
            x = F.relu(getattr(self, 'fc{}'.format(i))(x), inplace=True)
            if i in self.scales[1]:
                scale = self.scales[0, np.argwhere(i == self.scales[1])[0, 0]]
                logit = getattr(self, 'param_adj_cluster{}'.format(scale))(x)
                logit = getattr(self, 'fc_cluster{}'.format(scale))(logit).squeeze()
                Logit.append(logit)
        Logit = torch.cat(Logit, dim=1)
        return F.softmax(Logit, dim=1) + 1e-9 / Logit.shape[1]

    def loss(self, p_cluster, ActiveVox_List):
        '''
        logistic loss of the activate patches
        :param p_cluster: the probability of the cluster being active
        :param ActiveVox_List:
        :return:
        '''
        p_vert = torch.matmul(p_cluster, self.adj_cluster_all.to_dense())
        log_p = torch.log(p_vert)
        # log_p = torch.log(p_cluster)
        # log_p = torch.matmul(log_p, self.adj_cluster_all.to_dense())

        loss = 0.
        batch_size = len(ActiveVox_List)
        for batch in range(batch_size):
            loss -= log_p[batch, ActiveVox_List[batch]].mean() / batch_size

        loss = 0.
        batch_size = len(ActiveVox_List)
        nSource = self.adj_cluster_all.shape[-1]
        for batch in range(batch_size):
            n = ActiveVox_List[batch].shape[0]
            w1 = 1 / n
            w2 = 1 / (nSource - n)
            loss -= (log_p[batch, ActiveVox_List[batch]].sum() * (w1 + w2) - log_p[batch].sum() * w2) / batch_size

        return loss

    def reconstruct(self, p_cluster, B_tensor, L_tensor, TBFs_svd_tensor, prune_rate=0.9):
        '''
        Reconstruct the source signal or TBFs projection
        :param p_cluster: the probability of the cluster being active
        :param B_tensor:
        :param L_tensor:
        :param TBFs_svd_tensor:
        :param prune_rate:
        :return:
        '''
        # dtype = B_tensor.dtype
        # B_tensor = B_tensor.to(torch.float64)
        # L_tensor = L_tensor.to(torch.float64)
        # s_tensor = torch.zeros([B_tensor.shape[0], L_tensor.shape[-1], B_tensor.shape[-1]], dtype=torch.float64).to(self.device)
        # p_vert = torch.matmul(p_cluster, self.adj_cluster_all.to_dense())
        # p_max = torch.max(p_vert, dim=1, keepdim=True)[0]
        # batch_id, vert_id = torch.where(p_vert > p_max * prune_rate)
        # for batch in range(B_tensor.shape[0]):
        #     id = vert_id[torch.where(batch_id == batch)]
        #     s_tensor[batch, id] = torch.pinverse(L_tensor[batch, :, id]).matmul(B_tensor[batch])

        # p_max = torch.max(p_cluster, dim=1, keepdim=True)[0]
        # batch_id, cluster_id = torch.where(p_cluster > p_max * prune_rate)
        # for batch in range(B_tensor.shape[0]):
        #     id = cluster_id[torch.where(batch_id == batch)]
        #     source_ind = np.unique(np.hstack([self.cluster[i] for i in id]))
        #     s_tensor[batch, source_ind] = torch.pinverse(L_tensor[batch, :, source_ind]).matmul(B_tensor[batch])

        dtype = B_tensor.dtype
        B_tensor = B_tensor.to(torch.float64)
        L_tensor = L_tensor.to(torch.float64)
        TBFs_svd = TBFs_svd_tensor.pow(2)
        TBFs_svd = TBFs_svd / TBFs_svd.max(dim=-1, keepdim=True)[0]
        s_tensor = torch.zeros([B_tensor.shape[0], L_tensor.shape[-1], B_tensor.shape[-1]], dtype=torch.float64).to(
            self.device)
        # adj = self.adj_cluster_all.to_dense()
        # p_vert = torch.matmul(p_cluster, adj / adj.sum(dim=0, keepdim=True))
        p_vert = torch.matmul(p_cluster, self.adj_cluster_all.to_dense())
        p_max = torch.max(p_vert, dim=1, keepdim=True)[0]
        batch_id, vert_id = torch.where(p_vert > p_max * prune_rate)
        for batch in range(B_tensor.shape[0]):
            id = vert_id[torch.where(batch_id == batch)]
            L = L_tensor[batch, :, id]
            _, sv, _ = torch.svd(L, compute_uv=False)
            I = torch.eye(min(L.shape[0], L.shape[1])).to(self.device)
            for k in range(TBFs_svd.shape[-1]):
                sigma = TBFs_svd[batch, k] / sv.max()
                if L.shape[0] <= L.shape[1]:
                    p_inv = sigma * L.T.mm(torch.inverse(L.mm(L.T) * sigma + I))
                else:
                    p_inv = torch.inverse(L.T.mm(L) + I / sigma).mm(L.T)
                s_tensor[batch, id, k] = p_inv.matmul(B_tensor[batch, :, k])

        return s_tensor.to(dtype)


class Cluster_GCN_ADJ_new(BasicModule):
    '''
    Variational Graph Convolutional Network with Parameterized Adjacency Matrix Layer
    '''

    def __init__(self, k=5, dropout=0.5, Layers=[10], Cortex=None, Gain=None, device='cpu', perlayer=False):
        super(Cluster_GCN_ADJ_new, self).__init__()

        self.model_name = 'cgcn_adj'

        if Cortex is None:
            Cortex = cortex_loader()
        if Gain is None:
            Gain = gain_loader()

        self.scales = Cortex['Scales']

        cluster = []
        for i in range(len(Cortex['Clusters'])):
            scale = self.scales[i]
            cluster += Cortex['Clusters'][i]
            indices1 = np.hstack([np.full_like(Cortex['Clusters'][i][j], j) for j in range(len(Cortex['Clusters'][i]))])
            indices2 = np.hstack([Cortex['Clusters'][i][j] for j in range(len(Cortex['Clusters'][i]))])
            indices = torch.from_numpy(np.vstack([indices1, indices2]))
            values = torch.ones(indices.shape[1])
            shape = torch.Size([len(Cortex['Clusters'][i]), Gain.shape[1]])
            adj_cluster = torch.sparse_coo_tensor(indices, values, shape).coalesce().to(device)
            setattr(self, 'adj_cluster{}'.format(scale), adj_cluster)
            setattr(self, 'param_adj_cluster{}'.format(scale), ParameterizedAdj(adj_cluster))
            setattr(self, 'fc_cluster{}'.format(scale), nn.Linear(Layers[-1], 1))
        self.cluster = cluster

        indices1_all = np.hstack([np.full_like(cluster[i], i) for i in range(len(cluster))])
        indices2_all = np.hstack([cluster[i] for i in range(len(cluster))])
        indices_all = torch.from_numpy(np.vstack([indices1_all, indices2_all]))
        values_all = torch.ones(indices_all.shape[1])
        shape_all = torch.Size([len(cluster), Gain.shape[1]])
        self.adj_cluster_all = torch.sparse_coo_tensor(indices_all, values_all, shape_all).coalesce().to(device)

        VertConn = Cortex['VertConn']
        self.adj_vert = sparse_mx_to_torch_sparse_tensor(VertConn + sp.eye(VertConn.shape[0]),
                                                         normal=False).coalesce().to(device)
        self.adj_sensor = gain_to_sparse(Gain, normal=False, percentage=99).t().coalesce().to(device)

        self.Layers = Layers
        self.perlayer = perlayer
        self.device = device

        self.fc_senor = nn.Linear(k, Layers[0])
        self.parm_adj_sensor = ParameterizedAdj(self.adj_sensor)

        for i in range(len(Layers) - 1):
            setattr(self, 'fc{}'.format(i + 1), nn.Linear(Layers[i], Layers[i + 1]))
            if perlayer:
                setattr(self, 'param_adj{}'.format(i + 1), ParameterizedAdj(self.adj_vert))
            elif i == 0:
                self.param_adj = ParameterizedAdj(self.adj_vert)

        self.to(device)

    def forward(self, x):
        x = F.relu(self.parm_adj_sensor(self.fc_senor(x)))
        for i in range(len(self.Layers) - 1):
            i += 1
            if self.perlayer:
                x = getattr(self, 'param_adj{}'.format(i))(x)
            else:
                x = self.param_adj(x)
            x = F.relu(getattr(self, 'fc{}'.format(i))(x), inplace=True)
        Logit = []
        for i in range(self.scales.shape[0]):
            scale = self.scales[i]
            logit = getattr(self, 'param_adj_cluster{}'.format(scale))(x)
            logit = getattr(self, 'fc_cluster{}'.format(scale))(logit).squeeze()
            Logit.append(logit)
        Logit = torch.cat(Logit, dim=1)
        return F.log_softmax(Logit, dim=1)

    def loss(self, logp_cluster, ActiveVox_List):
        '''
        logistic loss of the activate patches
        :param logp_cluster: the probability of the cluster being active
        :param ActiveVox_List:
        :return:
        '''
        p_vert = torch.matmul(logp_cluster.exp(), self.adj_cluster_all.to_dense())
        log_p = torch.log(p_vert + 1e-20)
        # log_p = torch.log(p_cluster)
        # log_p = torch.matmul(logp_cluster, self.adj_cluster_all.to_dense())

        loss = 0.
        batch_size = len(ActiveVox_List)
        for batch in range(batch_size):
            loss -= log_p[batch, ActiveVox_List[batch]].mean() / batch_size

        # loss = 0.
        # batch_size = len(ActiveVox_List)
        # nSource = self.adj_cluster_all.shape[-1]
        # for batch in range(batch_size):
        #     n = ActiveVox_List[batch].shape[0]
        #     w1 = 1 / n
        #     w2 = 1 / (nSource - n)
        #     loss -= (log_p[batch, ActiveVox_List[batch]].sum() * (w1 + w2) - log_p[batch].sum() * w2) / batch_size

        return loss

    def reconstruct(self, logp_cluster, B_tensor, L_tensor, TBFs_svd_tensor, prune_rate=0.9):
        '''
        Reconstruct the source signal or TBFs projection
        :param p_cluster: the probability of the cluster being active
        :param B_tensor:
        :param L_tensor:
        :param TBFs_svd_tensor:
        :param prune_rate:
        :return:
        '''

        dtype = B_tensor.dtype
        B_tensor = B_tensor.to(torch.float64)
        L_tensor = L_tensor.to(torch.float64)
        TBFs_svd = TBFs_svd_tensor.pow(2)
        TBFs_svd = TBFs_svd / TBFs_svd.max(dim=-1, keepdim=True)[0]
        s_tensor = torch.zeros([B_tensor.shape[0], L_tensor.shape[-1], B_tensor.shape[-1]], dtype=torch.float64).to(
            self.device)
        p_vert = torch.matmul(logp_cluster.exp(), self.adj_cluster_all.to_dense())
        p_max = torch.max(p_vert, dim=1, keepdim=True)[0]
        batch_id, vert_id = torch.where(p_vert > p_max * prune_rate)
        for batch in range(B_tensor.shape[0]):
            id = vert_id[torch.where(batch_id == batch)]
            L = L_tensor[batch, :, id]
            _, sv, _ = torch.svd(L, compute_uv=False)
            I = torch.eye(min(L.shape[0], L.shape[1])).to(self.device)
            for k in range(TBFs_svd.shape[-1]):
                sigma = TBFs_svd[batch, k] / sv.max()
                if L.shape[0] <= L.shape[1]:
                    p_inv = sigma * L.T.mm(torch.inverse(L.mm(L.T) * sigma + I))
                else:
                    p_inv = torch.inverse(L.T.mm(L) + I / sigma).mm(L.T)
                s_tensor[batch, id, k] = p_inv.matmul(B_tensor[batch, :, k])

        return s_tensor.to(dtype)


class Vertices_GCN_ADJ(BasicModule):
    '''
    Variational Graph Convolutional Network with Parameterized Adjacency Matrix Layer
    '''

    def __init__(self, k=5, dropout=0.5, Layers=[10], Cortex=None, Gain=None, device='cpu', perlayer=False):
        super(Vertices_GCN_ADJ, self).__init__()

        self.model_name = 'vgcn_adj'

        if Cortex is None:
            Cortex = cortex_loader()
        if Gain is None:
            Gain = gain_loader()

        VertConn = Cortex['VertConn']
        self.adj_vert = sparse_mx_to_torch_sparse_tensor(VertConn + sp.eye(VertConn.shape[0]),
                                                         normal=False).coalesce().to(device)
        self.adj_sensor = gain_to_sparse(Gain, normal=False, percentage=99).t().coalesce().to(device)

        self.Layers = Layers
        self.perlayer = perlayer
        self.device = device

        self.fc_senor = nn.Linear(k, Layers[0])
        self.parm_adj_sensor = ParameterizedAdj(self.adj_sensor)

        for i in range(len(Layers)):
            if i == len(Layers) - 1:
                setattr(self, 'fc{}'.format(i + 1), nn.Linear(Layers[i], 1))
            else:
                setattr(self, 'fc{}'.format(i + 1), nn.Linear(Layers[i], Layers[i + 1]))
            if perlayer:
                setattr(self, 'param_adj{}'.format(i + 1), ParameterizedAdj(self.adj_vert))
            elif i == 0:
                self.param_adj = ParameterizedAdj(self.adj_vert)

        self.to(device)

    def forward(self, x):
        x = F.relu(self.parm_adj_sensor(self.fc_senor(x)))
        for i in range(len(self.Layers)):
            i += 1
            x = getattr(self, 'fc{}'.format(i))(x)
            if self.perlayer:
                x = getattr(self, 'param_adj{}'.format(i))(x)
            else:
                x = self.param_adj(x)
            if i < len(self.Layers):
                x = F.relu(x)
        x = x.squeeze()
        x = x / x.abs().max(dim=-1, keepdim=True)[0] * 4.6
        return torch.sigmoid(x) + 1e-9 / x.shape[1]

    def loss(self, p_vert, ActiveVox_List):
        '''
        logistic loss of the activate patches
        :param p_vert: the probability of the vertices being active
        :param ActiveVox_List:
        :return:
        '''
        log_p = torch.log(p_vert)

        # loss = 0.
        # w = 1e-3
        # batch_size = len(ActiveVox_List)
        # for batch in range(batch_size):
        #     loss -= ((1 + w) * log_p[batch, ActiveVox_List[batch]].sum() - w * log_p[batch].sum()) / batch_size

        loss = 0.
        batch_size = len(ActiveVox_List)
        nSource = self.adj_vert.shape[0]
        rate = 1
        for batch in range(batch_size):
            n = ActiveVox_List[batch].shape[0]
            w1 = 1 / n
            w2 = rate / (nSource - n)
            loss -= (log_p[batch, ActiveVox_List[batch]].sum() * (w1 + w2) - log_p[batch].sum() * w2) / batch_size

        return loss

    def reconstruct(self, p_vert, B_tensor, L_tensor, TBFs_svd_tensor, prune_rate=0.9):
        '''
        Reconstruct the source signal or TBFs projection
        :param p_vert: the probability of the vertices being active
        :param B_tensor:
        :param L_tensor:
        :param TBFs_svd_tensor:
        :param prune_rate:
        :return:
        '''

        dtype = B_tensor.dtype
        B_tensor = B_tensor.to(torch.float64)
        L_tensor = L_tensor.to(torch.float64)
        TBFs_svd = TBFs_svd_tensor.pow(2)
        TBFs_svd = TBFs_svd / TBFs_svd.max(dim=-1, keepdim=True)[0]
        s_tensor = torch.zeros([B_tensor.shape[0], L_tensor.shape[-1], B_tensor.shape[-1]], dtype=torch.float64).to(
            self.device)
        p_max = torch.max(p_vert, dim=1, keepdim=True)[0]
        batch_id, vert_id = torch.where(p_vert > p_max * prune_rate)
        for batch in range(B_tensor.shape[0]):
            id = vert_id[torch.where(batch_id == batch)]
            L = L_tensor[batch, :, id]
            _, sv, _ = torch.svd(L, compute_uv=False)
            I = torch.eye(min(L.shape[0], L.shape[1])).to(self.device)
            for k in range(TBFs_svd.shape[-1]):
                sigma = TBFs_svd[batch, k] / sv.max()
                if L.shape[0] <= L.shape[1]:
                    p_inv = sigma * L.T.mm(torch.inverse(L.mm(L.T) * sigma + I))
                else:
                    p_inv = torch.inverse(L.T.mm(L) + I / sigma).mm(L.T)
                s_tensor[batch, id, k] = p_inv.matmul(B_tensor[batch, :, k])

        return s_tensor.to(dtype)


class Vertices_GCN_ADJ2(BasicModule):
    '''
    Variational Graph Convolutional Network with Parameterized Adjacency Matrix Layer
    '''

    def __init__(self, k=5, dropout=0.5, Layers=[10], Cortex=None, Gain=None, device='cpu', perlayer=False):
        super(Vertices_GCN_ADJ2, self).__init__()

        self.model_name = 'vgcn_adj'

        if Cortex is None:
            Cortex = cortex_loader()
        if Gain is None:
            Gain = gain_loader()

        VertConn = Cortex['VertConn']
        self.adj_vert = sparse_mx_to_torch_sparse_tensor(VertConn + sp.eye(VertConn.shape[0]),
                                                         normal=False).coalesce().to(device)
        self.adj_sensor = gain_to_sparse(Gain, normal=False, percentage=90).t().coalesce().to(device)

        self.Layers = Layers
        self.perlayer = perlayer
        self.device = device

        self.fc_TBFs_in = nn.Linear(k, Layers[0], bias=False)
        self.parm_adj_sensor = ParameterizedAdj(self.adj_sensor)

        for i in range(len(Layers) - 1):
            setattr(self, 'fc{}'.format(i + 1), nn.Linear(Layers[i], Layers[i + 1]))
            if perlayer:
                setattr(self, 'param_adj{}'.format(i + 1), ParameterizedAdj(self.adj_vert))
            elif i == 0:
                self.param_adj = ParameterizedAdj(self.adj_vert)

        self.fc_TBFs_out = nn.Linear(Layers[0], k, bias=False)

        self.to(device)

    def forward(self, x):
        x = F.leaky_relu(self.parm_adj_sensor(self.fc_TBFs_in(x)), 0.2)
        for i in range(len(self.Layers) - 1):
            i += 1
            x = getattr(self, 'fc{}'.format(i))(x)
            if self.perlayer:
                x = getattr(self, 'param_adj{}'.format(i))(x)
            else:
                x = self.param_adj(x)
            if i < len(self.Layers) - 1:
                x = F.leaky_relu(x, 0.2)
        x = self.fc_TBFs_out(x)
        return x

    def loss(self, s_est, s_real):
        '''
        logistic loss of the activate patches
        :param s_est:
        :param s_real:
        :return:
        '''
        loss = (s_real - s_est).pow(2).sum(dim=[-1, -2]).mean()

        return loss

    def reconstruct(self, p_vert, B_tensor, L_tensor, TBFs_svd_tensor, prune_rate=0.9):
        '''
        Reconstruct the source signal or TBFs projection
        :param p_vert: the probability of the vertices being active
        :param B_tensor:
        :param L_tensor:
        :param TBFs_svd_tensor:
        :param prune_rate:
        :return:
        '''

        dtype = B_tensor.dtype
        B_tensor = B_tensor.to(torch.float64)
        L_tensor = L_tensor.to(torch.float64)
        TBFs_svd = TBFs_svd_tensor.pow(2)
        TBFs_svd = TBFs_svd / TBFs_svd.max(dim=-1, keepdim=True)[0]
        s_tensor = torch.zeros([B_tensor.shape[0], L_tensor.shape[-1], B_tensor.shape[-1]], dtype=torch.float64).to(
            self.device)
        p_max = torch.max(p_vert, dim=1, keepdim=True)[0]
        batch_id, vert_id = torch.where(p_vert > p_max * prune_rate)
        for batch in range(B_tensor.shape[0]):
            id = vert_id[torch.where(batch_id == batch)]
            L = L_tensor[batch, :, id]
            _, sv, _ = torch.svd(L, compute_uv=False)
            I = torch.eye(min(L.shape[0], L.shape[1])).to(self.device)
            for k in range(TBFs_svd.shape[-1]):
                sigma = TBFs_svd[batch, k] / sv.max()
                if L.shape[0] <= L.shape[1]:
                    p_inv = sigma * L.T.mm(torch.inverse(L.mm(L.T) * sigma + I))
                else:
                    p_inv = torch.inverse(L.T.mm(L) + I / sigma).mm(L.T)
                s_tensor[batch, id, k] = p_inv.matmul(B_tensor[batch, :, k])

        return s_tensor.to(dtype)


class Vertices_GAT(BasicModule):
    '''
    Graph Attention network Network for vertices
    '''

    def __init__(self, k=5, dropout=0.5, Layers=[10], nheads=8, Cortex=None, Gain=None, device='cpu'):
        super(Vertices_GAT, self).__init__()
        self.model_name = 'vgat'

        if Cortex is None:
            Cortex = cortex_loader()
        if Gain is None:
            Gain = gain_loader()

        self.Layers = Layers
        self.device = device

        self.adj_vert = sparse_mx_to_torch_sparse_tensor(Cortex['VertConn'], normal=True).coalesce().to(device)
        self.adj_sensor = gain_to_sparse(Gain, normal=True, percentage=90).t().coalesce().to(device)
        self.gc_senor = GraphConvolutionDense(k, Layers[0])
        self.fc_TBFs = nn.Linear(k, Layers[0], bias=False)

        for i in range(len(Layers) - 1):
            self.add_module('attentions_l{}'.format(i),
                            GraphAttentionLayer(Layers[i], Layers[i + 1], dropout=dropout, alpha=0.2, nheads=nheads,
                                                adj=self.adj_vert))
            if i < len(Layers) - 2:
                self.add_module('attention_l{}out'.format(i),
                                GraphAttentionLayer(Layers[i + 1] * nheads, Layers[i + 1], dropout=dropout, alpha=0.2,
                                                    nheads=1, adj=self.adj_vert))

        self.to(device)

    def forward(self, x):
        x = F.leaky_relu(self.gc_senor(x, self.adj_sensor))
        for i in range(len(self.Layers) - 1):
            x = getattr(self, 'attentions_l{}'.format(i))(x)
            if i < len(self.Layers) - 2:
                x = F.leaky_relu(getattr(self, 'attention_l{}out'.format(i))(x))
        x = x.matmul(self.fc_TBFs.weight)
        return x

    def loss(self, s_est, s_real):
        '''
        logistic loss of the activate patches
        :param s_est:
        :param s_real:
        :return:
        '''
        loss = (s_real - s_est).pow(2).sum(dim=[-1, -2]).mean()

        return loss


class Cluster_Attention_Network(BasicModule):
    def __init__(self, k=5, dropout=0.5, Layers=[10], Cortex=None, Gain=None, device='cpu'):
        super(Cluster_Attention_Network, self).__init__()
        self.model_name = 'can'

        if Cortex is None:
            Cortex = cortex_loader()
        if Gain is None:
            Gain = gain_loader()
        self.adj_vert = sparse_mx_to_torch_sparse_tensor(Cortex['VertConn'], normal=True).coalesce().to(device)
        adj_sensor = gain_to_sparse(Gain, normal=True, percentage=90).t().coalesce()
        indices = adj_sensor.indices()
        adj_temp = adj_sensor.to_dense()
        values = (adj_temp / adj_temp.sum(1, keepdim=True))[indices[0], indices[1]]
        self.adj_sensor = torch.sparse_coo_tensor(indices, values, adj_sensor.shape, device=device).coalesce()

        self.Layers = Layers

        self.gc_senor = GraphConvolutionDense(k, Layers[0], bias=False)
        # self.TFs_out = nn.Linear(Layers[-1], k, bias=False)

        self.clusters = []
        for i in range(len(Cortex['Clusters'])):
            self.clusters += Cortex['Clusters'][i]

        indices1_all = np.hstack([np.full_like(self.clusters[i], i) for i in range(len(self.clusters))])
        indices2_all = np.hstack([self.clusters[i] for i in range(len(self.clusters))])
        indices_all = torch.from_numpy(np.vstack([indices1_all, indices2_all]))
        values_all = torch.ones(indices_all.shape[1])
        shape_all = torch.Size([len(self.clusters), Gain.shape[1]])
        self.adj_cluster = torch.sparse_coo_tensor(indices_all, values_all, shape_all).coalesce().to(device)

        for i in range(len(Layers)):
            self.add_module('clusters_att{}'.format(i),
                            ClusterAttention2(self.clusters, self.adj_cluster, Layers[i], 200))
            if i < len(Layers) - 1:
                self.add_module('gc{}'.format(i), GraphConvolutionDense(Layers[i], Layers[i + 1]))

        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)

        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.gc_senor(x, self.adj_sensor)
        for i in range(len(self.Layers)):
            att_vert, att_clu = getattr(self, 'clusters_att{}'.format(i))(x)
            x = att_vert * x
            if i < len(self.Layers) - 1:
                x = getattr(self, 'gc{}'.format(i))(x, self.adj_vert)
        # x = x.matmul(self.gc_senor.weight.t())
        # return x
        return att_vert.squeeze(), att_clu.squeeze()

    # def loss(self, s_est, s_real):
    #     '''
    #     logistic loss of the activate patches
    #     :param s_est:
    #     :param s_real:
    #     :return:
    #     '''
    #     loss = (s_real - s_est).pow(2).sum(dim=[-1, -2]).mean()
    #
    #     return loss

    def loss(self, p_vert, ActiveVox_List):
        '''
        logistic loss of the activate patches
        :param logp_cluster: the probability of the cluster being active
        :param ActiveVox_List:
        :return:
        '''
        log_p = torch.log(p_vert + 1e-20)

        loss = 0.
        batch_size = len(ActiveVox_List)
        nSource = self.adj_vert.shape[0]
        for batch in range(batch_size):
            n = ActiveVox_List[batch].shape[0]
            w1 = 1 / n
            w2 = 1 / (nSource - n)
            loss -= (log_p[batch, ActiveVox_List[batch]].sum() * (w1 + w2) - log_p[batch].sum() * w2) / batch_size

        return loss

    def reconstruct(self, p_clusters, B_tensor, L_tensor, TBFs_tensor, TBFs_svd_tensor, prune_rate=0.1):
        '''
        Reconstruct the source signal or TBFs projection
        :param p_clusters: the probability of the clusters being active
        :param B_tensor:
        :param L_tensor:
        :param TBFs_svd_tensor:
        :param prune_rate:
        :return:
        '''
        dtype = B_tensor.dtype
        B_tensor = B_tensor.to(torch.float64)
        L_tensor = L_tensor.to(torch.float64)
        TBFs_tensor = TBFs_tensor.to(torch.float64)
        B_proj_tensor = B_tensor.bmm(TBFs_tensor.transpose(1, 2))
        TBFs_svd = TBFs_svd_tensor.pow(2)
        TBFs_svd = TBFs_svd / TBFs_svd.max(dim=-1, keepdim=True)[0]
        s_est = torch.zeros([B_tensor.shape[0], L_tensor.shape[-1], B_proj_tensor.shape[-1]], dtype=torch.float64).to(
            self.device)
        batch_id, clusters_id = torch.where(p_clusters > p_clusters.max(-1, keepdim=True)[0] * prune_rate)
        for batch in range(B_tensor.shape[0]):
            id_clu = clusters_id[torch.where(batch_id == batch)]
            id_vert = np.unique(np.hstack([self.clusters[id_clu[i]] for i in range(id_clu.shape[0])]))
            L = L_tensor[batch, :, id_vert]
            _, sv, _ = torch.svd(L, compute_uv=False)
            I = torch.eye(min(L.shape[0], L.shape[1])).to(self.device)
            for k in range(TBFs_svd.shape[-1]):
                sigma = TBFs_svd[batch, k] / sv.max()
                if L.shape[0] <= L.shape[1]:
                    p_inv = sigma * L.T.mm(torch.inverse(L.mm(L.T) * sigma + I))
                else:
                    p_inv = torch.inverse(L.T.mm(L) + I / sigma).mm(L.T)
                s_est[batch, id_vert, k] = p_inv.matmul(B_proj_tensor[batch, :, k])
        s_est = s_est.matmul(TBFs_tensor)
        return s_est.to(dtype)


class Vertices_Attention_Network(BasicModule):
    def __init__(self, k=5, dropout=0.5, Layers=[10], Cortex=None, Gain=None, device='cpu', perlay=True):
        super(Vertices_Attention_Network, self).__init__()
        self.model_name = 'van'

        if Cortex is None:
            Cortex = cortex_loader()
        if Gain is None:
            Gain = gain_loader()
        self.adj_vert = sparse_mx_to_torch_sparse_tensor(Cortex['VertConn'], normal=True).coalesce().to(device)
        adj_sensor = gain_to_sparse(Gain, normal=True, percentage=90).t().coalesce()
        indices = adj_sensor.indices()
        adj_temp = adj_sensor.to_dense()
        values = (adj_temp / adj_temp.sum(1, keepdim=True))[indices[0], indices[1]]
        self.adj_sensor = torch.sparse_coo_tensor(indices, values, adj_sensor.shape, device=device).coalesce()

        self.Layers = Layers

        self.gc_senor = GraphConvolutionDense(k, Layers[0], bias=False)

        att_features = 100
        for i in range(len(Layers)):
            if perlay:
                vertatt = VerticesAttention(self.adj_vert, Layers[i], att_features)
            else:
                if i == 0:
                    std = 1 / sqrt(att_features)
                    att = nn.Parameter(torch.full(size=(self.adj_vert.shape[0], att_features), fill_value=-std))
                    att.data.uniform_(-std, std)
                vertatt = VerticesAttention(self.adj_vert, Layers[i], att_features, att=att)
            self.add_module('vertices_att{}'.format(i), vertatt)
            if i < len(Layers) - 1:
                self.add_module('gc{}'.format(i), GraphConvolutionDense(Layers[i], Layers[i + 1]))

        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)

        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.gc_senor(x, self.adj_sensor)
        for i in range(len(self.Layers)):
            att_vert = getattr(self, 'vertices_att{}'.format(i))(x)
            x = att_vert * x
            if i < len(self.Layers) - 1:
                x = getattr(self, 'gc{}'.format(i))(x, self.adj_vert)
        # x = x.matmul(self.gc_senor.weight.t())
        # return x
        return att_vert.squeeze()

    def loss(self, p_vert, ActiveVox_List):
        '''
        logistic loss of the activate patches
        :param logp_cluster: the probability of the cluster being active
        :param ActiveVox_List:
        :return:
        '''
        log_p = torch.log(p_vert + 1e-20)

        loss = 0.
        batch_size = len(ActiveVox_List)
        nSource = self.adj_vert.shape[0]
        for batch in range(batch_size):
            n = ActiveVox_List[batch].shape[0]
            w1 = 1 / n
            w2 = 1 / (nSource - n)
            loss -= (log_p[batch, ActiveVox_List[batch]].sum() * (w1 + w2) - log_p[batch].sum() * w2) / batch_size

        return loss

    def reconstruct(self, p_vert, B_tensor, L_tensor, TBFs_tensor, TBFs_svd_tensor, prune_rate=0.1):
        '''
        Reconstruct the source signal or TBFs projection
        :param p_vert: the probability of the vertices being active
        :param B_tensor:
        :param L_tensor:
        :param TBFs_svd_tensor:
        :param prune_rate:
        :return:
        '''
        dtype = B_tensor.dtype
        B_tensor = B_tensor.to(torch.float64)
        L_tensor = L_tensor.to(torch.float64)
        TBFs_tensor = TBFs_tensor.to(torch.float64)
        B_proj_tensor = B_tensor.bmm(TBFs_tensor.transpose(1, 2))
        TBFs_svd = TBFs_svd_tensor.pow(2)
        TBFs_svd = TBFs_svd / TBFs_svd.max(dim=-1, keepdim=True)[0]
        s_est = torch.zeros([B_tensor.shape[0], L_tensor.shape[-1], B_proj_tensor.shape[-1]], dtype=torch.float64).to(
            self.device)
        batch_id, vert_id = torch.where(p_vert > p_vert.max(-1, keepdim=True)[0] * prune_rate)
        for batch in range(B_tensor.shape[0]):
            id_vert = vert_id[torch.where(batch_id == batch)]
            L = L_tensor[batch, :, id_vert]
            _, sv, _ = torch.svd(L, compute_uv=False)
            I = torch.eye(min(L.shape[0], L.shape[1])).to(self.device)
            for k in range(TBFs_svd.shape[-1]):
                sigma = TBFs_svd[batch, k] / sv.max()
                if L.shape[0] <= L.shape[1]:
                    p_inv = sigma * L.T.mm(torch.inverse(L.mm(L.T) * sigma + I))
                else:
                    p_inv = torch.inverse(L.T.mm(L) + I / sigma).mm(L.T)
                s_est[batch, id_vert, k] = p_inv.matmul(B_proj_tensor[batch, :, k])
        s_est = s_est.matmul(TBFs_tensor)
        return s_est.to(dtype)
