from scipy.io import loadmat
import os
import pickle
import scipy.sparse as sparse
import numpy as np

from utils import tess_area, create_clusters


def cortex_loader(path='Cortex_6003'):
    '''
    Load cortex structure from mat file in the dictionary form
    :param path: file path
    :return:
    '''
    if '.pkl' in path:
        path_pkl = path
    else:
        path_pkl = path + '.pkl'

    # 如果存在pkl文件则直接从pkl文件中读取Cortex结构体，否则从.mat文件中读取
    if os.path.exists(os.path.join('Cortex', path_pkl)):
        with open(os.path.join('Cortex', path_pkl), 'rb') as fo:  # 从pkl文件中读取Cortex结构体
            Cortex = pickle.load(fo)
            return Cortex
    else:
        if '.' in path:
            path = os.path.splitext(path)[0]
        path += '.mat'

        Cortex_temp = loadmat(os.path.join('Cortex', path))['Cortex'][0, 0]
        Cortex = dict()

        Cortex['Faces'] = Cortex_temp['Faces'] - 1
        Cortex['Vertices'] = Cortex_temp['Vertices']
        Cortex['VertConn'] = Cortex_temp['VertConn']
        Cortex['VertNormals'] = Cortex_temp['VertNormals']

        # calculate the Vertices Area and save it in the cortex dictionary
        _, Cortex['VertArea'] = tess_area(Cortex['Vertices'], Cortex['Faces'])

        # calculate the neighborhood of the vertices in 10 steps
        VertConn = Cortex['VertConn'].copy()
        VertConn = sparse.eye(VertConn.shape[0], VertConn.shape[1]) + VertConn
        Cortex['Neighborhood'] = []
        Cortex['Neighborhood'].append(VertConn)
        for i in range(9):
            Cortex['Neighborhood'].append(Cortex['Neighborhood'][i].dot(VertConn))

        # evaluate the basic clusters of the cortex and the corresponding spatial basis functions
        score = np.random.rand(VertConn.shape[0])
        scales = [3, 4, 5, 6]
        seeds = []
        clusters = []
        for i in range(len(scales)):
            seed_, _, cluster_ = create_clusters(Cortex, score, scales[i])
            seeds.append(np.array(seed_))
            clusters.append(cluster_)
        # clusters, seeds, scales = patch_expand(VertConn, [3, 4, 5])
        Cortex['Clusters'] = clusters
        Cortex['Seeds'] = seeds
        Cortex['Scales'] = np.array(scales)
        # Cortex['SBFs'] = []
        # for i in range(len(scales)):
        #     Cortex['SBFs'].append(SpatialBasisFunctions(VertConn, clusters[i]))
        #     print(scales[i])

        # 将Cortex结构体写入到pkl文件中
        with open(os.path.join('Cortex', path_pkl), 'wb') as f:
            pickle.dump(Cortex, f)

        return Cortex


def gain_loader(path='Gain_6003'):
    '''
    Load the Lead Field matrix from mat file in the numpy array form
    :param path:
    :return:
    '''
    if '.pkl' in path:
        path_pkl = path
    else:
        path_pkl = path + '.pkl'

    # 如果存在pkl文件则直接从pkl文件中读取Cortex结构体，否则从.mat文件中读取
    if os.path.exists(os.path.join('Cortex', path_pkl)):
        with open(os.path.join('Cortex', path_pkl), 'rb') as fo:  # 从pkl文件中读取Cortex结构体
            Gain = pickle.load(fo)
            return Gain
    else:
        if '.' in path:
            path = os.path.splitext(path)[0]
        path += '.mat'

        Gain = loadmat(os.path.join('Cortex', path))['Gain']

        # 将Cortex结构体写入到pkl文件中
        with open(os.path.join('Cortex', path_pkl), 'wb') as f:
            pickle.dump(Gain, f)

        return Gain
