import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix


def tess_area(Vertices, Faces):
    '''
    Compute the surface area associated with each face and each vertex.
    :param Vertices: Vertices array
    :param Faces: Faces array
    :return:
    '''

    r12 = Vertices[Faces[:, 0], :]
    r13 = Vertices[Faces[:, 2], :] - r12
    r12 = Vertices[Faces[:, 1], :] - r12

    # 通过计算向量叉乘求出三角形(Face)的面积
    FaceArea = np.sqrt(np.sum(np.power(np.cross(r12, r13), 2), axis=1)) / 2

    # 通过构建稀疏矩阵求出每个vertices对应的面积
    nFaces = Faces.shape[0]
    rowno = np.hstack([Faces[:, 0], Faces[:, 1], Faces[:, 2]])
    colno = np.hstack([np.arange(nFaces), np.arange(nFaces), np.arange(nFaces)])
    data = np.hstack([FaceArea, FaceArea, FaceArea])
    VertFacesArea = coo_matrix((data, (rowno, colno)))

    # 每个vertices分得每个Face面积的1/3
    VertArea = np.array(VertFacesArea.sum(axis=1) / 3)

    return FaceArea, VertArea


def tess_scout_swell(Verts, VertConn):
    '''
    Enlarge a patch by appending the next set of adjacent vertices.
    :param Verts: initial vertices in the patch
    :param VertConn: vertices adjacent sparse matrix
    :return: new vertices being appended
    '''

    verts_new = np.max(VertConn[Verts], axis=0).col
    verts_new = verts_new[~np.isin(verts_new, Verts)]

    return verts_new


def patch_generate(Seed, VertConn, VertArea, AreaDef):
    '''
    Generate a patch given the seeds and area extents of sources
    :param Seed: Seed Voxel
    :param VertConn: vertices adjacent sparse matrix
    :param VertArea: area of each vertices
    :param AreaDef: Area of each extended source associated with each seed voxel
    :return:
    '''

    Patch = Seed
    Area = np.sum(VertArea[Patch])
    # 不断扩大Patch，直到Patch中的vertices满足面积要求
    while Area <= AreaDef:
        if Area <= AreaDef:
            verts_new = tess_scout_swell(Patch, VertConn)
            if verts_new.shape[0] == 0:
                break
            else:
                Nouter = np.append(Patch, verts_new)
                Area = np.sum(VertArea[Nouter])
        if Area > AreaDef:
            Ndiff = Nouter[~np.isin(Nouter, Patch)]
            for i in range(len(Ndiff)):
                Patch = np.append(Patch, Ndiff[i])
                Area = np.sum(VertArea[Patch])
                if Area > AreaDef:
                    break
        else:
            Patch = Nouter

    return Patch


def active_vox_generator(seedvox, AreaDef, Cortex):
    '''
    Generate active voxel with given seed voxel, area of each extended source and cortex structure
    :param seedvox: Seed Voxel
    :param AreaDef: Area of each extended source associated with each seed voxel
    :param Cortex: Cortex structure
    :return:
    '''

    Patch = list()
    ActiveVox = np.empty(shape=[0], dtype=int)
    # _, VertArea = tess_area(Cortex['Vertices'], Cortex['Faces'])
    VertArea = Cortex['VertArea']
    for k in range(len(seedvox)):
        patch = patch_generate(seedvox[k], Cortex['VertConn'], VertArea, AreaDef[k])
        Patch.append(patch)
        ActiveVox = np.append(ActiveVox, patch)
    Area = np.sum(VertArea[ActiveVox])

    return Patch, ActiveVox, Area


def variation_edge(vertconn):
    '''
    generate the sparse matrix of the variation matrix
    :param vertconn:
    :return:
    '''
    nSource = vertconn.shape[0]
    row = np.empty([0], dtype=np.int)
    col = np.empty([0], dtype=np.int)
    indices = np.empty([0], dtype=np.int)
    for i in range(nSource):
        col_1, col_2, _ = sp.find(vertconn[i, :] != 0)
        idx = np.where(col_2 < i)[0]
        col_1 = col_1[idx] + i
        col_2 = col_2[idx]
        row_temp = (np.arange(0, idx.shape[0]) + row.shape[0] / 2).astype(np.int)
        row = np.hstack([row, row_temp, row_temp])
        col = np.hstack([col, col_1, col_2])
        indices = np.hstack([indices, np.full(idx.shape[0], 1), np.full(idx.shape[0], -1)])

    return coo_matrix((indices, (row, col)))


def create_clusters(Cortex, scores, extent):
    neighborhood = Cortex['Neighborhood'][extent - 1]
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
