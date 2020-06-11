from scipy import sparse
from scipy.linalg import cholesky
import math
import numpy as np


def spatial_prior(vertconn, rho=0.6):
    nSource = vertconn.shape[0]
    W = sparse.eye(nSource)
    if np.all(vertconn.diagonal() == 1):
        vertconn -= sparse.eye(nSource)

    A = vertconn - sparse.diags(vertconn.sum(axis=1).A.squeeze(), 0)
    A0 = rho * A / 2
    for i in range(7):
        W += A0
        A0 = rho * A.dot(A0) / (i + 2) / 2
    W.data[W.data < math.exp(-8)] = 0.
    W = W.dot(W.T)
    return W


def spatial_basis_functions(vertconn, clusters):
    W = spatial_prior(vertconn)

    for i in range(len(clusters)):
        Sigma = W[clusters[i], :]
        Sigma = Sigma[:, clusters[i]]
        Sigma = cholesky(Sigma.todense(), lower=True)
        Sigma = Sigma / np.linalg.norm(Sigma, axis=1)
        row = np.repeat(clusters[i].reshape(-1, 1), clusters[i].shape[0], 1).T.reshape(-1)
        col = np.repeat(np.arange(clusters[i].shape[0]), clusters[i].shape[0], 0)
        SBFs_temp = sparse.coo_matrix((Sigma.T.reshape(-1), (row, col)), shape=(W.shape[0], Sigma.shape[0]))
        if i == 0:
            SBFs = SBFs_temp
        else:
            SBFs = sparse.hstack([SBFs, SBFs_temp])

    return SBFs


def patch_expand(vertconn, scales):
    if type(scales) is not list:
        scales = [scales]
    scales = np.sort(np.array(scales))

    nSource = vertconn.shape[0]
    if np.all(vertconn.diagonal() == 0):
        vertconn += sparse.eye(nSource)

    neighborhood = sparse.eye(nSource)
    neighborhoods = [neighborhood]
    for i in range(max(scales)):
        neighborhood = neighborhoods[i].dot(vertconn)
        neighborhoods.append(neighborhood)

    Patch = []
    Seed = []
    for n in range(scales.shape[0]):
        scale = scales[n]
        patch_list = []
        seed_list = []
        seeds = np.array([np.random.randint(nSource)])
        interior = np.empty([0], dtype=np.int)
        n = 0
        while True:
            seeds_temp = np.empty([0], dtype=np.int)
            interior_temp = interior
            for i in range(seeds.shape[0]):
                if np.in1d(seeds[i], interior_temp):
                    continue
                n += 1
                seed_list.append(seeds[i])
                patch = np.unique(np.argwhere(neighborhoods[scale][seeds[i], :] != 0)[:, 1])
                patch_list.append(patch)
                inter = np.unique(np.argwhere(neighborhoods[scale - 1][seeds[i], :] != 0)[:, 1])
                interior_temp = np.append(interior_temp, inter)
                boundary = np.setdiff1d(patch, inter)
                seeds_temp = np.append(seeds_temp, boundary)
            seeds = np.setdiff1d(seeds_temp, interior_temp)
            interior = np.unique(interior_temp)
            if interior.shape[0] == nSource:
                print(n, scale)
                Patch.append(patch_list)
                Seed.append(np.array(seed_list, dtype=patch.dtype))
                break
            elif seeds.shape[0] == 0:
                ind = np.random.randint(nSource-interior.shape[0])
                seeds = np.array([np.setdiff1d(np.arange(nSource), interior)[ind]])

    return Patch, Seed, scales
