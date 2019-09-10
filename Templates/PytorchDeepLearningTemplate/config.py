import warnings
import torch


class DefaultConfig(object):
    '''
    定义默认设置的类，改变实验参数只须要调整config文件即可
    '''
    env = 'default'
    model = 'AlexNet'

    train_data_root = '/home2/liangjw/Documents/Pycharm_project/Pytorch_test/data/CatsAndDogs/train'
    test_data_root = '/home2/liangjw/Documents/Pycharm_project/Pytorch_test/data/CatsAndDogs/test1'
    load_model_path = None

    batch_size = 128
    use_gpu = True
    num_workers = 4
    print_freq = 20 # print info every N batch

    max_epoch = 10
    lr = 0.1 # initial learning rate
    lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4

    def _parse(self,kwargs):
        '''
        根据字典kwargs中的参数，调整config的参数
        '''
        for key,value in kwargs.items():
            if not hasattr(self, key):
                warnings.warn("Warning: opt has not attribut %s" % key)
            setattr(self, key, value)

        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')

        print('user config:')
        for key,value in self.__class__.__dict__.items():
            if not key.startswith('_'):
                print(key,':',getattr(self,key))

Config = DefaultConfig()
Config._parse({'lr':0.05})
print(Config.use_gpu)