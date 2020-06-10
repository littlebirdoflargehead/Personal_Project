import pickle
import os
import math
import numpy as np
import torch

from Cortex import gain_loader


def simulated_signal_loader(path=None):
    '''
    Load simulated signal from the pkl files
    :param path:
    :return:
    '''
    with open(path, 'rb') as fo:
        Data = pickle.load(fo)

    ActiveVox_List = Data['ActiveVox_List']
    s_real_List = Data['s_real_List']
    ratio_array = Data['ratio_array']
    B_array = Data['B_array']
    W_array = Data['W_array']
    W_singular_array = Data['W_singular_array']
    TBFs_array = Data['TBFs_array']
    TBFs_svd_array = Data['TBFs_svd_array']

    return ActiveVox_List, s_real_List, ratio_array, B_array, W_array, W_singular_array, TBFs_array, TBFs_svd_array


class simulated_signal_dataset(object):
    def __init__(self, path='', filters=[], gain=None, device='cpu'):

        dir_list = os.listdir(path)
        self.data_files = [os.path.join(path, n) for n in self.dir_filter(dir_list, filters)]
        self.path = path

        if len(self.data_files) == 0:
            raise ValueError('No data files')

        if gain is None:
            gain = gain_loader()
        self.gain = torch.from_numpy(gain.astype(np.float32)).to(device)
        self.device = device

        self.reset()

    def reset(self):
        '''
        Reset all the parameters of the buffer
        :return:
        '''
        self.read_files = []
        self.nbuffer = 0
        self.buffer_id = 0
        self.buffer_order = np.empty([0])
        self.buffer = {'ActiveVox_List': [],
                       's_real_List': [],
                       'ratio_array': np.empty([0]),
                       'B_array': np.empty([0, 0, 0]),
                       'W_array': np.empty([0, 0, 0]),
                       'W_singular_array': np.empty([0]),
                       'TBFs_array': np.empty([0, 0, 0]),
                       'TBFs_svd_array': np.empty([0, 0])}

    def dir_filter(self, dir_list=[], filters=[]):
        '''
        Return directory lists satisfied the filters' condition
        :param dir_list: directory lists
        :param filters:
        :return:
        '''
        data_files = []
        for name in dir_list:
            satisfied = True
            for f in filters:
                if f not in name:
                    satisfied = False
                    break
            if satisfied:
                data_files.append(name)

        return data_files

    def batch_generator_torch(self, batch_size=128, multi_files=False, nfiles=5):
        '''
        Generate batch samples for training or testing in torch format
        :param batch_size: number of samples in one batch
        :param multi_files: generate samples from multiple files or not(used for old version)
        :param nfiles: number of multiple files to load if multi_files is True
        :return:
        '''
        ActiveVox_List, s_real_List, ratio_array, B_array, W_array, W_singular_array, TBFs_array, TBFs_svd_array \
            = self.batch_generator(batch_size, nfiles)

        ratio_tensor, B_tensor, W_tensor, W_singular_tensor, TBFs_tensor, TBFs_svd_tensor = \
            self.np_to_torch(ratio_array, B_array, W_array, W_singular_array, TBFs_array, TBFs_svd_array)

        L_tensor = torch.matmul(W_tensor, self.gain)

        return ActiveVox_List, s_real_List, ratio_tensor, B_tensor, L_tensor, W_singular_tensor, TBFs_tensor, TBFs_svd_tensor

    def np_to_torch(self, *args):
        '''
        Translate numpy array to torch tensor
        :param args: array needed to be translated
        :return:
        '''
        out = []
        for n in range(len(args)):
            out.append(torch.from_numpy(args[n].astype(np.float32)).to(self.device))
        return out

    def torch_to_np(self, *args):
        '''
        Translate torch tensor to numpy array
        :param args: tensor needed to be translated
        :return:
        '''
        out = []
        for n in range(len(args)):
            out.append(args[n].detach().cpu().numpy().astype(np.float64))
        return out

    def batch_generator(self, batch_size=128, nfiles=2):
        '''
        Generate batch samples for training or testing in numpy format (new version with more memory but higher speed)
        :param batch_size: number of samples in one batch
        :param nfiles: number of files update into the buffer each time
        :return:
        '''
        # 当buffer为空时，初始化dataset，并对buffer进行更新
        if len(self.read_files) == 0:
            print('initialize the buffer')
            self.buffer_update(nfiles=nfiles)
            print('initialization finished')

        # 当buffer数量不足时，对其进行更新或对dataset进行重置
        if self.buffer_id + batch_size > self.nbuffer:
            if len(self.read_files) < len(self.data_files):
                print('update the buffer')
                self.buffer_update(nfiles=nfiles)
                print('update finished')
            elif len(self.read_files) == len(self.data_files):
                batch_size = self.nbuffer - self.buffer_id
                self.read_files = []

        # 从随机序列中抽取指定位置的样本
        data_index = self.buffer_order[self.buffer_id:self.buffer_id + batch_size]
        self.buffer_id += batch_size

        # 从buffer读取对应的样本并生成batch形式的张量
        ActiveVox_List = [self.buffer['ActiveVox_List'][i] for i in data_index]
        s_real_List = [self.buffer['s_real_List'][i] for i in data_index]
        ratio_array = self.buffer['ratio_array'][data_index]
        B_array = self.buffer['B_array'][data_index]
        W_array = self.buffer['W_array'][data_index]
        W_singular_array = self.buffer['W_singular_array'][data_index]
        TBFs_array = self.buffer['TBFs_array'][data_index]
        TBFs_svd_array = self.buffer['TBFs_svd_array'][data_index]

        return ActiveVox_List, s_real_List, ratio_array, B_array, W_array, W_singular_array, TBFs_array, TBFs_svd_array

    def buffer_update(self, nfiles):
        '''
        Select n data files to update into the buffer of this dataset object
        :param nfiles: number of files update into the buffer each time
        '''
        # 从未载入到buffer中的数据文件中随机抽取一定数目的数据文件进行加载
        unread_files = []
        for f in self.data_files:
            if f not in self.read_files:
                unread_files.append(f)
        if nfiles > len(unread_files):
            nfiles = len(unread_files)
        file_index = np.random.permutation(len(unread_files))[0:nfiles]

        # 对buffer中已读取的数据进行删除
        self.buffer['ActiveVox_List'] = self.buffer['ActiveVox_List'][self.buffer_id:self.nbuffer]
        self.buffer['s_real_List'] = self.buffer['s_real_List'][self.buffer_id:self.nbuffer]
        self.buffer['ratio_array'] = self.buffer['ratio_array'][self.buffer_id:self.nbuffer]
        self.buffer['B_array'] = self.buffer['B_array'][self.buffer_id:self.nbuffer]
        self.buffer['W_array'] = self.buffer['W_array'][self.buffer_id:self.nbuffer]
        self.buffer['W_singular_array'] = self.buffer['W_singular_array'][self.buffer_id:self.nbuffer]
        self.buffer['TBFs_array'] = self.buffer['TBFs_array'][self.buffer_id:self.nbuffer]
        self.buffer['TBFs_svd_array'] = self.buffer['TBFs_svd_array'][self.buffer_id:self.nbuffer]
        self.nbuffer -= self.buffer_id

        # 从未载入到buffer中的数据文件中随机抽取一定数目的数据文件进行加载
        for i in range(len(file_index)):
            ActiveVox_List, s_real_List, ratio_array, B_array, W_array, W_singular_array, TBFs_array, TBFs_svd_array \
                = simulated_signal_loader(unread_files[file_index[i]])
            self.buffer['ActiveVox_List'] += ActiveVox_List
            self.buffer['s_real_List'] += s_real_List
            self.buffer['W_singular_array'] = np.append(self.buffer['W_singular_array'], ratio_array)
            self.buffer['ratio_array'] = np.append(self.buffer['ratio_array'], ratio_array)
            self.nbuffer += len(ActiveVox_List)
            if len(self.read_files) == 0:
                self.buffer['B_array'] = B_array
                self.buffer['W_array'] = W_array
                self.buffer['TBFs_array'] = TBFs_array
                self.buffer['TBFs_svd_array'] = TBFs_svd_array
            else:
                self.buffer['B_array'] = np.concatenate((self.buffer['B_array'], B_array), axis=0)
                self.buffer['W_array'] = np.concatenate((self.buffer['W_array'], W_array), axis=0)
                self.buffer['TBFs_array'] = np.concatenate((self.buffer['TBFs_array'], TBFs_array), axis=0)
                self.buffer['TBFs_svd_array'] = np.concatenate((self.buffer['TBFs_svd_array'], TBFs_svd_array), axis=0)
            self.read_files.append(unread_files[file_index[i]])

        # 确定随机序列buffer_order，并初始化该序列的指针为0
        self.buffer_order = np.random.permutation(self.nbuffer)
        self.buffer_id = 0
