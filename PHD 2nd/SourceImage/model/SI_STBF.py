from math import pi, log
import time

import numpy as np
import scipy.sparse as sp

from Cortex import cortex_loader, gain_loader
from Cortex import spatial_basis_functions


class SISTBF(object):
    def __init__(self, Cortex=None, Gain=None):

        if Cortex is None:
            Cortex = cortex_loader()

        self.vertconn = Cortex['VertConn']
        self.neighbour = [self.vertconn + sp.eye(self.vertconn.shape[0])]

    def source_imaging(self, B, L, ratio, extent):
        seed, clusters = self.auto_cluster(B, L, extent)
        cls = np.array([c.shape[0] for c in clusters])
        SBFs = self.spatial_basis_functions(clusters)
        prune = [1e-6, 1e-1]
        max_iter = 1000
        s_est, param = self.variational_bayesian(B, L, SBFs, prune, cls, max_iter)
        s_est = np.array(s_est)*ratio
        param['clusters'] = [clusters[param['keeplistSBFs'][i]] for i in range(len(param['keeplistSBFs']))]

        return s_est, param

    def auto_cluster(self, B, L, extent):
        # Normalize Lead Field Matrix
        Gstruct = dict()
        Gn = L / np.sqrt(np.power(L, 2).sum(0))
        u, sv, _ = np.linalg.svd(Gn, full_matrices=False)
        Gstruct['Gn'] = Gn
        Gstruct['sv'] = sv
        Gstruct['u'] = u

        nSource = L.shape[1]
        t = B.shape[1]
        SCR = np.zeros((nSource, t))
        SVD_threshold = 0.95

        # MSP Calculation
        U, S, V = np.linalg.svd(B, full_matrices=False)

        inertia = np.cumsum(np.power(S, 2)) / np.power(S, 2).sum()
        q = np.where(inertia > SVD_threshold)[0][0] + 1

        APM_temp = np.zeros((nSource, q))
        scale_prob = np.max(np.abs(V[0:q]))

        for i in range(q):
            M_data = S[i] * U[:, i, np.newaxis].dot(V[np.newaxis, i, :])
            APM_temp[:, i] = self.msp(M_data, Gstruct)

        APM_tot = APM_temp.dot(V[0:q]) / scale_prob
        APM_tot = APM_tot / np.max(APM_tot)

        SCR = SCR + APM_tot - SCR * APM_tot

        seed = []
        cellstruct_cls = []

        for i in range(len(extent)):
            seed_, _, cellstruct_cls_ = self.create_clusters(np.mean(SCR, 1), extent[i])
            seed += seed_
            cellstruct_cls += cellstruct_cls_
        seed = np.array(seed)

        return seed, cellstruct_cls

    def variational_bayesian(self, B, L, SBFs, prune, cls, max_iter):
        t = time.time()
        # Initial of Algorithm
        F = L * SBFs.todense()
        nSensor, nSource = F.shape
        nSnap = B.shape[-1]
        epsilon = 1e-7
        beta = 1.
        C_noise = np.mat(np.eye(nSensor) / beta)
        update = 'Convex'
        FreeEnergyCompute = True
        Cost = [0.]

        K = 5
        _, _, D = np.linalg.svd(np.mat(B), full_matrices=False)
        Phi = D[0: K]
        theta = np.mat(np.zeros((nSource, K)))
        c = np.ones(K)
        Sigma_C = 1e-6 * np.mat(np.eye(K))

        clusters = np.insert(cls, 0, 0).cumsum()
        clusters = [np.arange(clusters[i], clusters[i + 1]) for i in range(len(cls))]
        clusters_temp = clusters

        FF = F
        ncls = len(clusters)
        keeplistTBFs = np.arange(K)  # Select the TBF hyperparameters
        keeplistSBFs = np.arange(ncls)  # Selcet the SBF hyperparameters
        alpha = 1e0 * (np.ones(ncls) + 1e-3 * np.random.randn(ncls)) * np.power(F, 2).sum() * np.power(Phi, 2).sum() / (
                nSnap * np.trace(C_noise))

        # variational_bayesian
        for iter in range(max_iter):
            # Temporal Basis Check
            if Phi.shape[0] > 0:
                index1 = np.argwhere(np.abs(1 / c) > np.abs(1 / c).max() * prune[1]).squeeze()
                index2 = np.argwhere(np.abs(1 / alpha) > np.abs(1 / alpha).max() * prune[0]).squeeze()
                try:
                    if index1.shape[0] < c.shape[0] or iter == 0:
                        c = c[index1]
                        keeplistTBFs = keeplistTBFs[index1]
                        Phi = Phi[index1]
                        K = index1.shape[0]
                except Exception as E:
                    print(E)
                if index2.shape[0] < alpha.shape[0] or iter == 0:
                    alpha = alpha[index2]
                    keeplistSBFs = keeplistSBFs[index2]
                    clusters = [clusters[index2[i]] for i in range(index2.shape[0])]
                    remaindip = np.hstack(clusters)
                    Ncls = np.array([c.shape[0] for c in clusters])
                    clusters_temp = np.insert(Ncls, 0, 0).cumsum()
                    clusters_temp = [np.arange(clusters_temp[i], clusters_temp[i + 1]) for i in range(len(Ncls))]
                    gamma = np.hstack([alpha[i] * np.ones(Ncls[i]) for i in range(len(Ncls))])
                    F = FF[:, remaindip]
                theta = theta[remaindip, :]
                theta = theta[:, index1]
            # W Update
            Diag_C = np.zeros(K)
            Diag_W = np.zeros(len(clusters))
            FAFT = np.multiply(F, np.repeat(1 / gamma[np.newaxis, :], nSensor, 0)) * F.T
            for k in range(K):
                x = Phi[k, :] * Phi[k, :].T + nSnap * Sigma_C[k, k]
                Sig_B = FAFT + C_noise / x
                Sig_B_inv = np.linalg.inv(Sig_B)
                index = np.setdiff1d(np.arange(K), k)
                residual = B * Phi[k, :].T - F * (
                        theta[:, index] * (Phi[index, :] * Phi[k, :].T + nSnap * Sigma_C[index, k]))
                theta[:, k] = np.multiply(np.repeat(1 / gamma[:, np.newaxis], nSensor, 1),
                                          F.T) * Sig_B_inv * residual / x
                Diag_C[k] = np.trace(FAFT * Sig_B_inv / x)
            # TBFs Update
            temp = F * theta
            C_noise_inv = np.linalg.inv(C_noise)
            Sigma_C = np.linalg.inv(temp.T * C_noise_inv * temp + np.diag(Diag_C + c))
            Phi = Sigma_C * temp.T * C_noise_inv * B
            # C Update
            c = 1 / (np.diag(Sigma_C) + np.diag(Phi * Phi.T) / nSnap)
            # alpha Update
            mu = np.zeros(alpha.shape[0])
            X = Phi * Phi.T + nSnap * Sigma_C
            inverse_temp = []
            for i in range(len(clusters)):
                col = clusters_temp[i]
                if update == 'Convex':
                    for k in range(K):
                        x = X[k, k]
                        if len(inverse_temp) > k:
                            inverse = inverse_temp[k]
                        else:
                            inverse = np.linalg.inv(FAFT + np.eye(F.shape[0]) / x)
                            inverse_temp.append(inverse)
                        mu[i] += np.trace(F[:, col].T * inverse * F[:, col])
                    alpha[i] = 1 / np.sqrt(np.trace(theta[col] * theta[col].T) / mu[i])
            # Recover theta
            temp = np.mat(np.zeros((nSource, K)))
            temp[remaindip] = theta
            theta = temp
            # Free Energy Computation
            if FreeEnergyCompute:
                gamma = np.hstack([alpha[i] * np.ones(Ncls[i]) for i in range(len(Ncls))])
            if update == 'Convex':
                cost = -0.5 * np.trace(B.T * C_noise_inv * B) + np.trace(
                    Phi.T * np.linalg.inv(Sigma_C) * Phi) + 0.5 * nSnap * (np.log(c).sum()) + np.log(
                    np.linalg.det(Sigma_C)) - 0.5 * np.trace(
                    np.multiply(theta[remaindip].T, np.repeat(1 / gamma[np.newaxis, :], K, 0)) * theta[remaindip])
                FAFT = np.multiply(F, np.repeat(1 / gamma[np.newaxis, :], nSensor, 0)) * F.T
                for k in range(K):
                    x = X[k, k]
                    cost += -0.5 * np.log(np.linalg.det(x * FAFT * C_noise_inv + np.eye(nSensor)))
                temp = F * theta[remaindip]
                temp = temp.T * C_noise_inv * temp
                cost += -0.5 * np.trace(Phi.T * temp * Phi) - 0.5 * nSnap * np.trace(
                    temp * Sigma_C) - 0.5 * nSnap * nSensor * log(2 * pi / beta)
            # Check stop conditon
            MSE = (cost - Cost[iter]) / cost
            Cost.append(cost)
            if abs(MSE) < epsilon:
                break
            print('iter = {}, MSE = {},  #TBFs = {}, #SBFs = {}, #remaindipols = {}'.format(iter, MSE, K, len(clusters),
                                                                                            F.shape[1]))
            print(1 / c)
        t = time.time() - t
        param = dict()
        param['Phi'] = Phi
        param['remaindip'] = remaindip
        param['theta'] = theta[remaindip]
        param['SBFs'] = SBFs.todense()[:, remaindip]
        param['keeplistSBFs'] = keeplistSBFs
        param['runtime'] = t
        s_est = param['SBFs'] * param['theta'] * param['Phi']

        return s_est, param

    def spatial_basis_functions(self, clusters):
        SBFs = spatial_basis_functions(self.vertconn, clusters)
        return SBFs

    def msp(self, M, Gstruct):
        MSP_R2_threshold = 0.92
        scores = np.zeros((Gstruct['Gn'].shape[1], 1))

        Mn = M / np.sqrt(np.power(M, 2).sum(0))
        Mn[np.isnan(Mn)] = 0
        gamma = Gstruct['u'].T.dot(Mn)
        R2 = np.diag(gamma.dot(gamma.T))

        indices = np.argsort(R2)[::-1]
        sv = Gstruct['sv'][indices]
        i_T = indices[1: np.where(sv.cumsum() / sv.sum() > MSP_R2_threshold)[0][0]]
        Ut = Gstruct['u'][:, np.sort(i_T)]

        Ms = Ut.dot(Ut.T).dot(Mn)
        _, S, V = np.linalg.svd(Ms, full_matrices=False)
        tol = np.sqrt(max(Ms.shape) * np.finfo(S.dtype).eps)
        r = np.where(S > tol)[0][-1] + 1
        S = S[np.newaxis, 0:r]
        V = V[0:r]
        S = 1 / np.power(S, 2)
        temp = Ms.dot(V.T)
        Ps = temp.dot(np.repeat(S, r, axis=0)).dot(temp.T)
        scores = np.sum(Ps.dot(Gstruct['Gn']) * Gstruct['Gn'], 0)

        return scores

    def create_clusters(self, scores, extent):
        if len(self.neighbour) < extent:
            while True:
                self.neighbour.append(self.neighbour[-1].dot(self.neighbour[0]))
                if len(self.neighbour) == extent:
                    break
        neighborhood = self.neighbour[extent - 1]
        nSource = scores.shape[0]

        indices = np.argsort(scores)[::-1]
        # sorted_scores = scores[indices]

        ii = 0
        thresh_index = nSource
        selected_source = np.zeros(nSource, dtype=np.int)
        cluster_no = 1
        seed = []
        while ii < thresh_index:
            node = indices[ii]
            if selected_source[node] == 0:
                neighbors = np.argwhere(neighborhood[node] != 0)[:, 1]
                neighbors = neighbors[selected_source[neighbors] == 0]
                if neighbors.shape[0] >= 5:
                    selected_source[neighbors] = cluster_no
                    cluster_no += 1
                    seed.append(node)
            ii += 1

        free_nodes = indices[selected_source[indices[0:thresh_index]] == 0]
        while free_nodes.shape[0] > 0:
            for i in range(free_nodes.shape[0]):
                free_node = free_nodes[0]
                neighbors = np.argwhere(neighborhood[free_node] != 0)[:, 1]
                neighbors = neighbors[selected_source[neighbors] != 0]
                if neighbors.shape[0] > 0:
                    cluster_no = np.min(selected_source[neighbors])
                    selected_source[free_node] = cluster_no
                    free_nodes = np.setdiff1d(free_nodes, free_node)

        cellstruct = []
        for i in range(selected_source.max()):
            cellstruct.append(np.argwhere(selected_source == i + 1).squeeze())

        return seed, selected_source, cellstruct
