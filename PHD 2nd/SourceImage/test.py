import torch
import numpy as np
import pickle
import time
import os
import csv
import concurrent.futures

from Cortex import cortex_loader, gain_loader
from utils import simulated_signal_generator, basic_signal
from utils import unbiased_auc, se, mse, dle, sd
from data import simulated_signal_dataset
from model import GCN_ADJ, Cluster_GCN_ADJ, Vertices_GCN_ADJ, SISTBF, Vertices_Attention_Network

Cortex = cortex_loader('Cortex_6003')
Gain = gain_loader('Gain_6003')

nSensor, nSource = Gain.shape

Basis, StimTime = basic_signal()

k = 5

nPatches = [2]

extents = [6e-4, 16e-4]

SNR = [0, 5, 10, 15]

batch_size = 16

max_epoch = 100

device = torch.device("cpu")
######################################
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
######################################

si_stbf = SISTBF(Cortex)
######################################
# model = Vertices_Attention_Network(k=Basis.shape[-1], Layers=[128, 32, 32, 32, 32], Cortex=Cortex, Gain=Gain,
#                                    device=device)
# model.load(os.path.join('checkpoints', 'van-200621_21:21:59.pth'))
######################################

RecordFile = open(os.path.join('checkpoints', 'SourceImaging.csv'), 'a')
RecordWriter = csv.writer(RecordFile)


# Fields = ['NumberofPatches', 'Extents', 'SNR', 'Method', 'epoch', 'AUC', 'SD', 'DLE', 'SE', 'MSE']
# RecordWriter.writerow(Fields)


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


for patches in nPatches:
    for extent in extents:
        for snr in SNR:
            dataset = simulated_signal_dataset(path='data', gain=Gain, device=device, random=False,
                                               filters=['.pkl', 'test',
                                                        'snr{}_nPatches{}_extent{}'.format(snr, patches, extent)])
            ######################################
            # 当使用神经网络训练时不进行白化，否则进行白化
            ActiveVox_List, s_real_List, ratio_array, B_array, L_array, W_array, W_singular_array, TBFs_array, TBFs_svd_array \
                = dataset.batch_generator(batch_size=max_epoch, nfiles=1, whiten=True)
            ######################################
            s_real_array = source_reconstruct(ActiveVox_List, s_real_List, TBFs_array)
            index = 0
            for epoch in range(max_epoch):
                s_est_array, param = si_stbf.source_imaging(B_array[epoch], L_array[epoch], ratio_array[epoch], [3, 4, 5, 6])
                ######################################
                # if epoch == index:
                #     if batch_size + index > max_epoch:
                #         batch_size = max_epoch - index
                #     id = np.arange(index, index + batch_size)
                #     ratio_tensor, B_tensor, L_tensor, TBFs_tensor, TBFs_svd_tensor = \
                #         dataset.np_to_torch(ratio_array[id], B_array[id], L_array[id], TBFs_array[id],
                #                             TBFs_svd_array[id])
                #     p_vert = model(B_tensor)
                #     s_est = model.reconstruct(p_vert, B_tensor, L_tensor, TBFs_tensor, TBFs_svd_tensor,
                #                               prune_rate=0.2) * ratio_tensor.view(-1, 1, 1)
                #     s_est = dataset.torch_to_np(s_est)[0]
                #     index += batch_size
                # s_est_array = s_est[epoch-index+batch_size]
                ######################################
                AUC = unbiased_auc(s_real_array[epoch], s_est_array, Cortex=Cortex)
                DLE = dle(s_real_array[epoch], s_est_array, Cortex=Cortex)
                SD = sd(s_real_array[epoch], s_est_array, Cortex=Cortex)
                MSE = mse(s_real_array[epoch], s_est_array)
                SE = se(s_real_array[epoch], s_est_array)
                epoch += 1
                print('epoch {} snr{} nPatches{} extent{} finished'.format(epoch, snr, patches, extent))
                ######################################
                RecordWriter.writerow((patches, extent*1e4, snr, 'VAN', epoch, AUC, SD, DLE, SE, MSE))
                ######################################
                RecordFile.flush()

RecordFile.close()

# batch_size = len(ActiveVox_List)
# AUC = np.zeros(batch_size)
# DLE = np.zeros(batch_size)
# SD = np.zeros(batch_size)
# MSE = np.zeros(batch_size)
# SE = np.zeros(batch_size)
#
# t0 = time.time()
# for i in range(batch_size):
#     s_est, param = si_stbf.source_imaging(B_array[i], L_array[i], ratio_array[i], [3, 4, 5, 6])
#     print('epoch {} finished'.format(i))
#     AUC[i] = unbiased_auc(s_real[i], s_est, Cortex=Cortex)
#     DLE[i] = dle(s_real[i], s_est, Cortex=Cortex)
#     SD[i] = sd(s_real[i], s_est, Cortex=Cortex)
#     MSE[i] = mse(s_real[i], s_est)
#     SE[i] = se(s_real[i], s_est)
# t1 = time.time()
# print(t1 - t0)
#
# res = []
# with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
#     futures = []
#     for i in range(batch_size):
#         futures.append(executor.submit(si_stbf.source_imaging, B_array[i], L_array[i], ratio_array[i], [3, 4, 5, 6]))
#     for i, fut in enumerate(futures):
#         res.append(fut.result())
#         print('epoch {} finished'.format(i))
# t2 = time.time()
# print(t2 - t1)
