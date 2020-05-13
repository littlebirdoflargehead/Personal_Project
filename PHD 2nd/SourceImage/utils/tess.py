import numpy as np
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