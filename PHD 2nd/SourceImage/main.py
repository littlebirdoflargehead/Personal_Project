import torch
import numpy as np
import pickle
import time
import os
import csv

from Cortex import cortex_loader, gain_loader
from utils import simulated_signal_generator, basic_signal
from utils import auc, unbiased_auc, se, mse, dle, sd
from data import simulated_signal_dataset
from model import GCN, GCN_d

# =============================================================================
# import cortex data from .mat files or .pkl files
# =============================================================================

Cortex = cortex_loader('Cortex_6003')
Gain = gain_loader('Gain_6003')

nSensor, nSource = Gain.shape

k = 5

nPatches = 1

extent = 6e-4

SNR = 5

batch_size = 64

max_epoch = 200

Basis, StimTime = basic_signal()

device = 'cuda:0'

checkpoint_fre = 10


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

        Data = {'ActiveVox_List': ActiveVox_List,
                's_real_List': s_real_List,
                'ratio_array': ratio_array,
                'B_array': B_array,
                'W_array': W_array,
                'W_singular_array': W_singular_array,
                'TBFs_array': TBFs_array}

        for n in range(samples):
            print(n)

            'Generate Simulated EEG Data'
            ActiveVox, s_real, ratio, B, _, W, TBFs = simulated_signal_generator(
                nPatches=nPatches, extent=extent, nTBFs=k, basic=Basis,
                stim=StimTime, snr=SNR, gain=Gain, cortex=Cortex)

            W_svd = np.linalg.svd(W, compute_uv=False)

            'save the data in directory'
            ActiveVox_List.append(ActiveVox)
            s_real_List.append(s_real[ActiveVox, :])
            ratio_array[n] = ratio
            B_array[n, :, :] = B
            W_array[n, :, :] = W
            W_singular_array[n] = np.mean(W_svd)
            TBFs_array[n, :, :] = TBFs

        with open(os.path.join('data', "Data_{}_epoch{}_nPatches{}_extent{}.pkl".format(mode, epoch, nPatches, extent)),
                  'wb') as fo:
            pickle.dump(Data, fo)


def train():
    # GCN模型
    gcnd = GCN_d(k=k, Layers=[16, 32, 64, 32, 16], Cortex=Cortex, Gain=Gain, device=device)
    gcnd.load(os.path.join('checkpoints', 'gcn-200513_00:48:48.pth'))
    # 数据集(训练集与测试集)
    dataset_train = simulated_signal_dataset(path='data', gain=Gain, device=device,
                                             filters=['.pkl', 'train', '_nPatches{}'.format(nPatches),
                                                      '_extent{}'.format(extent)])
    dataset_test = simulated_signal_dataset(path='data', gain=Gain, device=device,
                                            filters=['.pkl', 'test', '_nPatches{}'.format(nPatches),
                                                     '_extent{}'.format(extent)])
    # 优化器
    optimizer = torch.optim.Adam(gcnd.parameters(), weight_decay=1e-4)
    optimizer.load_state_dict(torch.load(os.path.join('checkpoints', 'adam-200513_00:48:48.pth')))
    # 训练结果写入csv
    testFields = ['epoch', 'AUC', 'DLE', 'SD', 'MSE', 'SE', 'loss']
    testF = open(os.path.join('checkpoints', 'test.csv'), 'w')
    testW = csv.writer(testF)
    testW.writerow(testFields)

    _, gain_svd, _ = torch.svd(dataset_train.gain)

    for epoch in range(max_epoch):
        # 从训练集中读取数据进行训练
        iter = 0
        while True:
            t0 = time.time()
            ActiveVox_List, s_real_List, ratio_tensor, B_tensor, L_tensor, W_singular_tensor, TBFs_tensor = \
                dataset_train.batch_generator_torch(batch_size=batch_size, nfiles=2)
            L_svd = W_singular_tensor*gain_svd.mean()

            t1 = time.time()

            B_proj_tensor = B_tensor.bmm(TBFs_tensor.transpose(1, 2))
            s_proj_tensor = gcnd(B_proj_tensor)
            B_reproj_tensor = L_tensor.bmm(s_proj_tensor)

            s_ = s_proj_tensor.clone()
            for i in range(len(ActiveVox_List)):
                s = np.unique(np.argwhere(Cortex['VertConn'][ActiveVox_List[i], :] != 0)[:, 1])
                s_[i, s, :] = 0.

            loss = torch.mean((B_proj_tensor - B_reproj_tensor).pow(2)) + torch.sum((L_svd.view(-1,1,1)*s_).pow(2))  # torch.mean(L_tensor.bmm(s_).pow(2))
            # loss = torch.sum(s_.pow(2))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t2 = time.time()

            iter += 1
            print('epoch:{}  iteration:{} loss:{} 数据读取时间:{} 模型训练时间:{}'.format(epoch, iter, loss.item(), t1 - t0, t2 - t1))
            if len(dataset_train.read_files) == 0:
                break

        # 从测试集中读取数据进行测试
        if epoch % checkpoint_fre == checkpoint_fre-1:
            AUC= DLE= SD= MSE= SE=0.
            dataset_test.reset()
            ActiveVox_List, s_real_List, ratio_array, B_array, W_array, _, TBFs_array = \
                dataset_test.batch_generator(batch_size=batch_size, nfiles=1)
            ratio_tensor, B_tensor, W_array, TBFs_tensor = dataset_test.np_to_torch(ratio_array, B_array, W_array,
                                                                                    TBFs_array)
            L_tensor = torch.matmul(W_array, dataset_test.gain)
            B_proj_tensor = B_tensor.bmm(TBFs_tensor.transpose(1, 2))
            s_proj_tensor = gcnd(B_proj_tensor)
            B_reproj_tensor = L_tensor.bmm(s_proj_tensor)
            loss = torch.mean((B_proj_tensor - B_reproj_tensor).pow(2))
            s_est = s_proj_tensor.bmm(TBFs_tensor).detach().cpu().numpy().astype(np.float64)*ratio_array.reshape(-1,1,1)
            for i in range(len(ActiveVox_List)):
                s_real = np.zeros([nSource, TBFs_array.shape[-1]])
                s_real[ActiveVox_List[i], :] = s_real_List[i]
                AUC += unbiased_auc(s_real, s_est[i, :, :], Cortex=Cortex, act_ind=ActiveVox_List[i]) / batch_size
                DLE += dle(s_real, s_est[i, :, :], Cortex=Cortex) / batch_size
                SD += sd(s_real, s_est[i, :, :], Cortex=Cortex) / batch_size
                MSE += mse(s_real, s_est[i, :, :]) / batch_size
                SE += se(s_real, s_est[i, :, :]) / batch_size

            # 写入csv
            testW.writerow((epoch, AUC, DLE, SD, MSE, SE, loss.item()))
            testF.flush()

            # 保存模型参数以及优化器参数
            name = gcnd.save()
            p1, p2 = os.path.split(name)
            p21, p22 = p2.split('-')
            torch.save(optimizer.state_dict(), os.path.join(p1, 'adam-' + p22))

    testF.close()


# file_generator(epochs=1, samples=1000, train=False)
train()
