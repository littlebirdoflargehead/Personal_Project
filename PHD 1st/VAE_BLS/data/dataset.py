import os
from PIL import Image
from torch.utils import data
import torchvision.transforms as T


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