import os
import torch
import torchvision
from PIL import Image
from torch.utils import data
import torchvision.transforms as T
import pickle


class Sub_MNIST(data.Dataset):
    '''
    从MNIST中获取指定的label的数据集对象
    '''
    def __init__(self,root,transforms=None,train=True,sublabel=[0],numperlabel=None):

        if not transforms:
            self.transforms = torchvision.transforms.ToTensor()
        else:
            self.transforms = transforms
        dataset = torchvision.datasets.MNIST(root=root,train=train,download=False)

        for i in range(len(sublabel)):
            subdata = dataset.data[dataset.targets==sublabel[i]]
            subtargets = dataset.targets[dataset.targets==sublabel[i]]
            if numperlabel:
                index = torch.randperm(subdata.shape[0])[:numperlabel]
                subdata = subdata[index]
                subtargets = subtargets[index]
            if i==0:
                self.data = subdata
                self.label = subtargets
            else:
                self.data = torch.cat([self.data,subdata],dim=0)
                self.label = torch.cat([self.label,subtargets],dim=0)

    def __getitem__(self, index):
        data = self.transforms(self.data[index].numpy())
        label = self.label[index]
        return data,label

    def __len__(self):
        return self.data.shape[0]




class FreyFaces(data.Dataset):

    def __init__(self, root, transforms=None):

        if not transforms:
            self.transforms = torchvision.transforms.ToTensor()
        else:
            self.transforms = transforms

        f = open(os.path.join(root,'Freyfaces','freyfaces.pkl'), 'rb')
        x = pickle.load(f, encoding='latin1')
        self.data = torch.from_numpy(1-x).view(-1,28,20)

    def __getitem__(self, index):
        data = self.data[index]
        return data

    def __len__(self):
        return self.data.shape[0]




class GoodOrBadCloth(data.Dataset):
    '''
    数据集中获取好与坏的布
    '''
    def __init__(self,root,transforms=None,good=True,train=True):
        '''
        区分好布与坏布并获取文件路径
        '''
        if good:
            root = os.path.join(root, 'good')
            imgs = [os.path.join(root, img) for img in os.listdir(root)]
        else:
            imgs = []
            root = os.path.join(root, 'bad')
            for types in os.listdir(root):
                roots = os.path.join(root,types)
                img = [os.path.join(roots, img) for img in os.listdir(roots)]
                imgs.extend(img)
        self.imgs = imgs

        if not transforms:
            self.transforms = torchvision.transforms.ToTensor()
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        '''
        返回一张图片的数据
        '''
        img_path = self.imgs[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        return data

    def __len__(self):
        return self.imgs




class DogCat(data.Dataset):

    def __init__(self,root,transforms=None,train=True,test=False):
        '''
        目标：获取所有图片地址，并根据训练、验证、测试划分数据
        不要将读取图片等高强度计算的操作放在__init__中
        '''
        self.test = test

        # 获取图片的文件路径
        imgs = [os.path.join(root,img) for img in os.listdir(root)]

        # 对图片进行排序
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        # 划分训练、验证集，验证:训练 = 3:7
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7*imgs_num)]
        else:
            self.imgs = imgs[int(0.7*imgs_num):]

        if transforms==None:
            normalize = T.Normalize(mean=[0.485,0.456,0.406],
                                    std=[0.229,0.224,0.225])

            # 测试集和验证集
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            # 训练集
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        '''
        返回一张图片的数据
        对于测试集，没有label，返回图片id，如1000.jpg返回1000
        '''
        img_path = self.imgs[index]
        if self.test:
            label = int(img_path.split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data,label

    def __len__(self):
        return len(self.imgs)