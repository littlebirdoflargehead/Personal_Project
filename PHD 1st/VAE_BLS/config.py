import warnings
import torch
import math


class DefaultConfig(object):
    '''
    定义默认设置的类，改变实验参数只须要调整config文件即可
    '''
    env = 'default'
    model = 'VAE_BLS'

    train_data_root = '/home2/liangjw/Documents/Pycharm_project/Pytorch_test/data'
    test_data_root = '/home2/liangjw/Documents/Pycharm_project/Pytorch_test/data'
    load_model_path = None

    batch_size = 1000
    use_gpu = True
    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    num_workers = 4
    print_freq = 150 # print info every N batch

    max_epoch = 20
    lr = 0.001 # initial learning rate
    lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4

    N1 = 10  # feature nodes per window
    N2 = 10  # number of windows of feature nodes
    N3 = 500 # number of enhancement nodes
    M1 = 50  # number of adding enhance nodes per step

    image_per_row = int(8)
    total_images = int(math.pow(image_per_row,2))


    def _parse(self,kwargs):
        '''
        根据字典kwargs中的参数，调整config的参数
        '''
        for key,value in kwargs.items():
            if not hasattr(self, key):
                warnings.warn("Warning: opt has not attribut %s" % key)
            setattr(self, key, value)

        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')
        self.total_images = int(math.pow(self.image_per_row, 2))

        print('user config:')
        for key,value in self.__class__.__dict__.items():
            if not key.startswith('_'):
                print(key,':',getattr(self,key))

Config = DefaultConfig()