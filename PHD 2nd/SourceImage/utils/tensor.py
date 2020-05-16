import torch
import numpy as np
import scipy.sparse as sp


def sparse_mx_to_torch_sparse_tensor(sparse_mx, normal=True):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if normal:
        sparse_mx = normalize(sparse_mx)

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    """Row-normalize sparse matrix"""
    if np.all(mx.diagonal() == 0):
        mx += sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def gain_to_sparse(gain, percentage=95, normal=False, istorch=True):
    '''
    Generate sparse Adjacency Matrix from the Gain Matrix according to the coefficients
    :param gain: Gain(Lead Field) Matrix
    :param percentage:
    :param normal:
    :param istorch: return torch tensor or scipy sparse coo_matrix
    :return:
    '''
    if percentage < 1.:
        percentage *= 100

    if normal:
        gain = normalize(gain)

    g_abs = np.abs(gain)
    index = np.argwhere(g_abs > np.percentile(g_abs, q=percentage, axis=1).reshape(-1, 1))

    if istorch:
        indices = torch.from_numpy(index.T)
        values = torch.ones(indices.shape[1])
        shape = torch.Size(gain.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    else:
        return sp.coo_matrix((np.ones(index.shape[0]), (index[:, 0], index[:, 1])), shape=gain.shape)
