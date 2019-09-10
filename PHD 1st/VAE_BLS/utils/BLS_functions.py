import numpy as np
import torch



def shrinkage(a, b):
    '''
    压缩参数
    :param a:
    :param b:
    :return:
    '''
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z



def sparse_bls(A, b):
    '''
    稀疏化参数
    :param A: 输出结果
    :param b: 输入
    :return: 经过稀疏化后的参数
    '''
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk



def pinv(A, reg=2**-30):
    '''
    计算矩阵伪逆
    :param A: 须要计算伪逆的矩阵
    :param reg: 正则项系数
    :return: 矩阵伪逆
    '''
    # A_inv = torch.inverse(torch.t(A).mm(A) + reg * torch.eye(A.shape[1])).mm(torch.t(A))

    A = A.detach().numpy()
    Q,R = np.linalg.qr(A)
    A_inv = np.linalg.solve(R,Q.T)
    # A_inv = np.linalg.solve(reg * np.eye(A.shape[1]) + A.T.dot(A), A.T)
    return torch.from_numpy(A_inv)


def iter_pinv(A,A_inv,H,c=2**-30):
    '''
    使用迭代公式计算矩阵的伪逆
    :param A: 原矩阵
    :param A_inv: 原矩阵伪逆
    :param H: 增加的矩阵块
    :return: 更新后的矩阵及其伪逆
    '''
    D = A_inv.mm(H)
    C = H - A.mm(D)
    if torch.allclose(C, torch.tensor([0]).double()):
        B = pinv(D, 1).mm(A_inv)
    else:
        B = pinv(C,c)
    A_inv = torch.cat([A_inv-D.mm(B),B],dim=0)
    A = torch.cat([A,H],dim=1)
    return A,A_inv