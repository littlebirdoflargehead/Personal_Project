import numpy as np
import matplotlib.pyplot as plt


def auc(source_sim, source_est, act_ind=None, plot=False):
    '''
    Plot ROC(receiver operating characteristic) curve and compute AUC(Area under the ROC curve)
    :param source_sim: Simulated source
    :param source_est: Estimated source
    :param act_ind: Index of active vertices source
    :param plot: plot the ROC curve or not
    :return:
    '''
    q_sim = np.sum(np.power(source_sim, 2), axis=1)
    q_est = np.sum(np.power(source_est, 2), axis=1)
    source_act = np.zeros(q_sim.shape[0])

    if act_ind is None:
        source_act[q_sim > 0] = 1
    else:
        source_act[act_ind] = 1

    n_positive = np.sum(source_act)
    n_negative = q_sim.shape[0] - n_positive
    source_act_sorted = source_act[q_est.argsort()[::-1]]

    TPR = np.cumsum(source_act_sorted) / n_positive
    FPR = np.cumsum(1 - source_act_sorted) / n_negative
    AUC = np.sum(np.diff(FPR) * TPR[0:TPR.shape[0] - 1])

    if plot:
        plt.plot(FPR, TPR)
        plt.title('AUC={}'.format(AUC))
        plt.show()

    return AUC


def unbiased_auc(source_sim, source_est, Cortex, act_ind=None):
    '''
    Compute the unbiased AUC with the average of close AUC and far AUC
    :param source_sim: Simulated source
    :param source_est: Estimated source
    :param Cortex: Cortex structure
    :param act_ind: Index of active vertices source
    :return:
    '''
    # 从Cortex字典中读取neighborhood稀疏矩阵，默认10阶邻接矩阵
    neighborhood = Cortex['Neighborhood']

    if act_ind is None:
        act_ind = np.where(np.any(source_sim != 0, axis=1))[0]

    # 将所有vertices分成Close与Far两个集合
    _, Close = np.where(neighborhood[act_ind, :].toarray() > 0)
    Close = np.unique(Close)
    Far = np.setdiff1d(np.arange(source_sim.shape[0]), Close)
    Close = np.setdiff1d(Close, act_ind)

    # 分别在Close与Far两个集合上计算修正后的AUC并平均
    if Close.shape[0] == 0 or Far.shape[0] == 0:
        AUC = auc(source_sim, source_est, act_ind=act_ind)
    else:
        n = act_ind.shape[0]
        AUC_Close = 0.
        AUC_Far = 0.
        for i in range(50):
            close_ind = np.append(act_ind, np.random.permutation(Close)[0:n])
            AUC_Close += auc(source_sim[close_ind, :], source_est[close_ind, :]) / 50
            far_ind = np.append(act_ind, np.random.permutation(Far)[0:n])
            AUC_Far += auc(source_sim[far_ind, :], source_est[far_ind, :]) / 50
        AUC = (AUC_Close + AUC_Far) / 2

    return AUC


def mse(source_sim, source_est):
    '''
    evaluate the Mean Squared Error
    :param source_sim: Simulated source
    :param source_est: Estimated source
    :return:
    '''
    MSE = np.sum(np.power(source_sim - source_est, 2)) / np.sum(np.power(source_sim, 2))

    return MSE


def se(source_sim, source_est):
    '''
    evaluate the the Shape Error
    :param source_sim: Simulated source
    :param source_est: Estimated source
    :return:
    '''
    source_sim_norm = np.sqrt(np.sum(np.power(source_sim, 2)))
    source_est_norm = np.sqrt(np.sum(np.power(source_est, 2)))
    source_sim_normalized = source_sim / source_sim_norm
    source_est_normalized = source_est / source_est_norm

    SE = np.sum(np.power(source_sim_normalized - source_est_normalized, 2))

    return SE


def dle(source_sim, source_est, Cortex):
    '''
    evaluate the distance of localization error
    :param source_sim: Simulated source
    :param source_est: Estimated source
    :param Cortex: Cortex structure
    :return:
    '''
    q_sim = np.sum(np.power(source_sim, 2), axis=1)
    q_est = np.sum(np.power(source_est, 2), axis=1)
    Vertices = Cortex['Vertices']

    act_sim = q_sim > 0
    act_est = q_est > 0
    vert_sim = Vertices[act_sim, :]
    vert_est = Vertices[act_est, :]
    q_est = q_est / np.sum(q_est)
    q_est = q_est[act_est]

    # vertices间的距离矩阵，单位为m
    dist = np.zeros([vert_sim.shape[0], vert_est.shape[0]])
    for i in range(dist.shape[0]):
        dist[i, :] = np.sum(np.power(vert_est - vert_sim[i, :].reshape(1, -1), 2), axis=1)
    dist = np.sqrt(dist)

    # 找出离每个vertices距离最近的激活的voxel
    act_closet_ind = np.argmin(dist, axis=0)
    ind_unique = np.unique(act_closet_ind)

    # 对每个激活的voxel找到该源中能量最高的vertex的距离(以该点为定位点)，并对每个源求平均值
    DLE = 0.
    for k in ind_unique:
        ind_temp = act_closet_ind == k
        dist_temp = dist[k, ind_temp]
        i = np.argmax(q_est[ind_temp])
        DLE += dist_temp[i]
    DLE /= ind_unique.shape[0]

    return DLE


def sd(source_sim, source_est, Cortex):
    '''
    evaluate the spatial dispersion
    :param source_sim: Simulated source
    :param source_est: Estimated source
    :param Cortex: Cortex structure
    :return:
    '''
    q_sim = np.sum(np.power(source_sim, 2), axis=1)
    q_est = np.sum(np.power(source_est, 2), axis=1)
    Vertices = Cortex['Vertices']

    act_sim = q_sim > 0
    act_est = q_est > 0
    vert_sim = Vertices[act_sim, :]
    vert_est = Vertices[act_est, :]
    q_est = q_est / np.sum(q_est)
    q_est = q_est[act_est]

    # vertices间的距离矩阵，单位为m（此处未开根号）
    dist = np.zeros([vert_sim.shape[0], vert_est.shape[0]])
    for i in range(dist.shape[0]):
        dist[i, :] = np.sum(np.power(vert_est - vert_sim[i, :].reshape(1, -1), 2), axis=1)

    # 找出离每个vertices距离最近的激活的voxel
    act_closet_ind = np.argmin(dist, axis=0)
    ind_unique = np.unique(act_closet_ind)

    # 以能量作为权值，对每个vertex到该源最近激活点的距离求加权平均值
    SD = 0.
    for k in ind_unique:
        SD += np.sum(dist[k, act_closet_ind == k] * q_est[act_closet_ind == k])
    SD = np.sqrt(SD)

    return SD
