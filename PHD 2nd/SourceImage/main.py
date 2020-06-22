import torch
import numpy as np
import pickle
import time
import os
import csv

from Cortex import cortex_loader, gain_loader
from utils import simulated_signal_generator, basic_signal
from utils import auc, unbiased_auc, se, mse, dle, sd
from utils import variation_edge, sparse_mx_to_torch_sparse_tensor
from data import simulated_signal_dataset
from model import Cluster_GCN_ADJ, Cluster_GCN_ADJ_new, Vertices_GCN_ADJ2, Vertices_GAT, Cluster_Attention_Network, Vertices_Attention_Network

Cortex = cortex_loader('Cortex_6003')
Gain = gain_loader('Gain_6003')

nSensor, nSource = Gain.shape

k = 5

nPatches = 3

extent = 6e-4

SNR = 5

batch_size = 32

max_epoch = 100

Basis, StimTime = basic_signal()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")


checkpoint_fre = 1


# variation = sparse_mx_to_torch_sparse_tensor(variation_edge(Cortex['VertConn']), normal=False).to(device)


def file_generator(epochs=2, samples=500, train=True):
    if train:
        mode = 'train'
    else:
        mode = 'test'

    for epoch in range(epochs):

        ActiveVox_List = []
        s_real_List = []
        ratio_array = np.empty(samples)
        B_array = np.empty([samples, nSensor, Basis.shape[-1]])
        W_array = np.empty([samples, nSensor, nSensor])
        W_singular_array = np.empty(samples)
        TBFs_array = np.empty([samples, k, Basis.shape[-1]])
        TBFs_svd_array = np.empty([samples, k])
        TBFs_whiten_array = np.empty([samples, k, Basis.shape[-1]])
        TBFs_svd_whiten_array = np.empty([samples, k])

        Data = {'ActiveVox_List': ActiveVox_List,
                's_real_List': s_real_List,
                'ratio_array': ratio_array,
                'B_array': B_array,
                'W_array': W_array,
                'W_singular_array': W_singular_array,
                'TBFs_array': TBFs_array,
                'TBFs_svd_array': TBFs_svd_array,
                'TBFs_whiten_array': TBFs_whiten_array,
                'TBFs_svd_whiten_array': TBFs_svd_whiten_array
                }

        for n in range(samples):
            print(n)

            'Generate Simulated EEG Data'
            ActiveVox, s_real, ratio, B, W, TBFs, TBFs_svd, TBFs_whiten, TBFs_svd_whiten = simulated_signal_generator(
                nPatches=nPatches, extent=extent, nTBFs=k, basic=Basis, stim=StimTime, snr=SNR, gain=Gain,
                cortex=Cortex)

            W_svd = np.linalg.svd(W, compute_uv=False)

            'save the data in directory'
            ActiveVox_List.append(ActiveVox)
            s_real_List.append(s_real[ActiveVox, :])
            ratio_array[n] = ratio
            B_array[n] = B
            W_array[n] = W
            W_singular_array[n] = np.mean(W_svd)
            TBFs_array[n] = TBFs
            TBFs_svd_array[n] = TBFs_svd
            TBFs_whiten_array[n] = TBFs_whiten
            TBFs_svd_whiten_array[n] = TBFs_svd_whiten

        with open(os.path.join('data',
                               "Data_{}_epoch{}_snr{}_nPatches{}_extent{}.pkl".format(mode, epoch, SNR, nPatches,
                                                                                      extent)),
                  'wb') as fo:
            pickle.dump(Data, fo)


def train():
    # GCN模型
    # gcn = Vertices_GCN_ADJ2(k=Basis.shape[-1], Layers=[64, 64, 64, 64, 64], Cortex=Cortex, Gain=Gain, perlayer=True, device=device)
    # gcn = Cluster_Attention_Network(k=Basis.shape[-1], Layers=[128, 32, 32, 32, 32], Cortex=Cortex, Gain=Gain, device=device)
    gcn = Vertices_Attention_Network(k=Basis.shape[-1], Layers=[128, 32, 32, 32, 32], Cortex=Cortex, Gain=Gain, device=device, perlay=True)
    # gcn = Vertices_GAT(k=Basis.shape[-1], Layers=[64, 8, 8, 8, 8], nheads=8, Cortex=Cortex, Gain=Gain, device=device)
    # gcn.load(os.path.join('checkpoints', 'can-200621_07:57:13.pth'))
    # 数据集(训练集与测试集)
    dataset_train = simulated_signal_dataset(path='data', gain=Gain, device=device,
                                             filters=['.pkl', 'train'])
    dataset_test = simulated_signal_dataset(path='data', gain=Gain, device=device,
                                            filters=['.pkl', 'test'])
    # 优化器
    optimizer = torch.optim.Adam(gcn.parameters(), weight_decay=1e-4)
    # optimizer.load_state_dict(torch.load(os.path.join('checkpoints', 'adam-200621_07:57:13.pth')))
    # 训练结果写入csv
    trainFields = ['epoch', 'loss']
    trainF = open(os.path.join('checkpoints', '{}_train.csv').format(gcn.model_name), 'w')
    trainW = csv.writer(trainF)
    trainW.writerow(trainFields)
    testFields = ['epoch', 'AUC', 'DLE', 'SD', 'MSE', 'SE', 'loss']
    testF = open(os.path.join('checkpoints', '{}_test.csv').format(gcn.model_name), 'w')
    testW = csv.writer(testF)
    testW.writerow(testFields)

    # _, gain_svd, _ = torch.svd(dataset_train.gain)

    for epoch in range(max_epoch):
        # 从训练集中读取数据进行训练
        iter = 0
        loss_ = 0.
        samples = 0
        while True:
            t0 = time.time()
            ActiveVox_List, s_real_List, ratio_array, B_array, L_array, W_array, W_singular_array, TBFs_array, \
            TBFs_svd_array = dataset_train.batch_generator(batch_size=batch_size, nfiles=8, whiten=False)
            if len(ActiveVox_List) == 0:
                dataset_train.reset()
                break
            ratio_tensor, B_tensor, L_tensor, TBFs_tensor, TBFs_svd_tensor = \
                dataset_train.np_to_torch(ratio_array, B_array, L_array, TBFs_array, TBFs_svd_array)
            s_real_tensor = source_reconstruct(ActiveVox_List, s_real_List, TBFs_array, numpy=False)
            # L_svd = W_singular_tensor * gain_svd.mean()
            t1 = time.time()

            loss, s_est = Loss(gcn, ActiveVox_List, ratio_tensor, B_tensor, L_tensor, TBFs_tensor,
                               TBFs_svd_tensor, s_real_tensor=s_real_tensor, loss_only=True)
            # s_real_array = source_reconstruct(ActiveVox_List, s_real_List, TBFs_array)
            # s_est_array = dataset_train.torch_to_np(s_est)[0]
            # s_est = dataset_train.torch_to_np(s_tensor * ratio_tensor.view(-1, 1, 1))[0]
            # s_est = dataset_train.torch_to_np(s_proj_tensor.bmm(TBFs_tensor) * ratio_tensor.view(-1, 1, 1))[0]
            # AUC, DLE, SD, MSE, SE = assessment(s_real_array, s_est_array, mean_only=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t2 = time.time()

            iter += 1
            print(
                'epoch:{}  iteration:{} loss:{} 数据读取时间:{} 模型训练时间:{}'.format(epoch, iter, loss.item(), t1 - t0, t2 - t1))
            samples += len(ActiveVox_List)
            loss_ += len(ActiveVox_List) * loss.item()
            if len(dataset_train.read_files) == 0:
                break

        trainW.writerow((epoch, loss_ / samples))
        trainF.flush()

        # 从测试集中读取数据进行测试
        if (epoch + 1) % checkpoint_fre == 0:
            with torch.no_grad():
                ActiveVox_List, s_real_List, ratio_array, B_array, L_array, W_array, W_singular_array, TBFs_array, \
                TBFs_svd_array = dataset_test.batch_generator(batch_size=batch_size, nfiles=8, whiten=False)
                ratio_tensor, B_tensor, L_tensor, TBFs_tensor, TBFs_svd_tensor = \
                    dataset_train.np_to_torch(ratio_array, B_array, L_array, TBFs_array, TBFs_svd_array)

                s_real_array = source_reconstruct(ActiveVox_List, s_real_List, TBFs_array)
                s_real_tensor = source_reconstruct(ActiveVox_List, s_real_List, TBFs_array, numpy=False)

                loss, s_est = Loss(gcn, ActiveVox_List, ratio_tensor, B_tensor, L_tensor, TBFs_tensor,
                                   TBFs_svd_tensor, s_real_tensor=s_real_tensor, loss_only=False)
                s_est_array = dataset_test.torch_to_np(s_est)[0]
                # s_est = dataset_test.torch_to_np(s_proj_tensor.bmm(TBFs_tensor) * ratio_tensor.view(-1, 1, 1))[0]
            AUC, DLE, SD, MSE, SE = assessment(s_real_array, s_est_array)

            dataset_test.reset()

            # 写入csv
            testW.writerow((epoch, AUC, DLE, SD, MSE, SE, loss.item()))
            testF.flush()

        # 保存模型参数以及优化器参数
        if (epoch + 1) % (checkpoint_fre * 5) == 0:
            name = gcn.save()
            p1, p2 = os.path.split(name)
            p21, p22 = p2.split('-')
            torch.save(optimizer.state_dict(), os.path.join(p1, 'adam-' + p22))

    trainF.close()
    testF.close()


def Loss(gcn, ActiveVox_List, ratio_tensor, B_tensor, L_tensor, TBFs_tensor, TBFs_svd_tensor, s_real_tensor=None,
         loss_only=True):
    p_vert = gcn(B_tensor)
    loss = gcn.loss(p_vert, ActiveVox_List)
    # s_est = gcn(B_tensor)
    # loss = gcn.loss(s_est, s_real_tensor / ratio_tensor.view(-1, 1, 1))
    if loss_only:
        s_est = None
    else:
        # s_est = gcn.reconstruct(p_clu, B_tensor, L_tensor, TBFs_tensor, TBFs_svd_tensor, prune_rate=0.2)*ratio_tensor.view(-1,1,1)
        s_est = gcn.reconstruct(p_vert, B_tensor, L_tensor, TBFs_tensor, TBFs_svd_tensor, prune_rate=0.2) * ratio_tensor.view(-1, 1, 1)
        # s_est = s_est * ratio_tensor.view(-1, 1, 1)

    return loss, s_est


def source_reconstruct(ActiveVox_List, s_real_List, TBFs_array, numpy=True):
    if not (len(ActiveVox_List) == len(s_real_List) == TBFs_array.shape[0]):
        raise ValueError('batch size not match')
    id0 = np.hstack([np.full(ActiveVox_List[i].shape[0], i) for i in range(len(ActiveVox_List))])
    id1 = np.hstack(ActiveVox_List)
    s_real = np.concatenate(s_real_List, axis=0)
    if numpy:
        s_real_array = np.zeros([len(ActiveVox_List), nSource, TBFs_array.shape[-1]])
        s_real_array[id0, id1, :] = s_real
        return s_real_array
    else:
        s_real_tensor = torch.zeros(size=(len(ActiveVox_List), nSource, TBFs_array.shape[-1]), device=device)
        s_real_tensor[id0, id1, :] = torch.from_numpy(s_real.astype(np.float32)).to(device)
        return s_real_tensor


def assessment(s_real, s_est, mean_only=True):
    if not (s_real.shape[0] == s_est.shape[0]):
        raise ValueError('batch size not match')

    batch_size = s_real.shape[0]
    AUC = np.zeros(batch_size)
    DLE = np.zeros(batch_size)
    SD = np.zeros(batch_size)
    MSE = np.zeros(batch_size)
    SE = np.zeros(batch_size)
    for i in range(batch_size):
        AUC[i] = unbiased_auc(s_real[i], s_est[i], Cortex=Cortex)
        DLE[i] = dle(s_real[i], s_est[i], Cortex=Cortex)
        SD[i] = sd(s_real[i], s_est[i], Cortex=Cortex)
        MSE[i] = mse(s_real[i], s_est[i])
        SE[i] = se(s_real[i], s_est[i])

    if mean_only:
        AUC = np.mean(AUC)
        DLE = np.mean(DLE)
        SD = np.mean(SD)
        MSE = np.mean(MSE)
        SE = np.mean(SE)

    return AUC, DLE, SD, MSE, SE


for extent in [6e-4, 16e-4]:
    for SNR in [0, 5, 10, 15]:
        file_generator(epochs=3, samples=1000, train=True)
        file_generator(epochs=1, samples=1000, train=False)
# train()
