import torch
import torch.nn as nn
import time


class BasicModule(nn.Module):
    '''
    封装nn.Module，加入model_name属性，并提供save和load两个方法，方便保存和读取模型
    '''

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self))

    def save(self,name=None):
        '''
        保存模型，默认使用“模型名字+ 时间”作为文件名
        如
        '''
        if name==None:
            prefix = 'checkpoints/' + self.model_name + '-'
            name = time.strftime(prefix + '%y%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(),name)
        return name

    def load(self,path):
        '''
        从文件路径中读取模型的称呼
        '''
        self.load_state_dict(torch.load(path))