import keras
import keras.layers
from keras.layers import Conv2D,Activation,Deconv2D,PReLU
from keras.models import Model

import numpy as np



class FSRCNN:
    def __init__(self, img_width=96, img_height=96, batch_size=16, channels=3, scale=4 ,d=56, s=12,m=4):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.channels = channels
        self.scale = scale
        self.d = d
        self.s = s
        self.m = m

    def create_sr_model(self, ip):

        'Feature extraction'
        x = self.Conv_layers(self.d,5,ip)
        x = PReLU(shared_axes=[1,2])(x)
        'Shrinking'
        x = self.Conv_layers(self.s,1,x)
        x = PReLU(shared_axes=[1,2])(x)
        'Non-linear mapping'
        for i in range(self.m):
            x = self.Conv_layers(self.s,3,x)
        x = PReLU(shared_axes=[1,2])(x)
        'Expanding'
        x = self.Conv_layers(self.d,1,x)
        x = PReLU(shared_axes=[1,2])(x)
        'Deconvolution'
        outputs = Deconv2D(self.channels,kernel_size=9,strides=self.scale,padding='same')(x)

        '构成模型'
        model = Model(inputs=ip,outputs=outputs)

        return model


    def Conv_layers(self,figers,kenel_size,ip):
        x = Conv2D(figers,
                   kernel_size=kenel_size,
                   strides=1,
                   padding='same',
                   kernel_initializer=keras.initializers.random_normal(stddev=np.sqrt(2 / 9 /self.img_height)),
                   bias_initializer=keras.initializers.constant(0),
                   kernel_regularizer=keras.regularizers.l2(0.0001),
                   bias_regularizer=keras.regularizers.l2(0.0001))(ip)
        return x



class VDSR:
    def __init__(self, img_width=96, img_height=96, batch_size=16, channels=3, kernel_size=3, filters=64, dept=20):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.channels = channels
        self.kernel_size = kernel_size
        self.filters = filters
        self.dept = dept

    def create_sr_model(self, ip):

        '卷积层连接'
        x = self.Conv_layers(self.filters,ip)
        for d in range(self.dept-2):
            x = self.Conv_layers(self.filters,x)
            x = Activation('relu')(x)
        x = self.Conv_layers(self.channels,x)
        '残差连接'
        outputs = keras.layers.add([ip,x])

        '构成模型'
        model = Model(inputs=ip,outputs=outputs)

        return model


    def Conv_layers(self,figers,ip):
        x = Conv2D(figers,
                   kernel_size=self.kernel_size,
                   strides=1,
                   padding='same',
                   kernel_initializer=keras.initializers.random_normal(stddev=np.sqrt(2 / 9 /self.img_height)),
                   bias_initializer=keras.initializers.constant(0),
                   kernel_regularizer=keras.regularizers.l2(0.0001),
                   bias_regularizer=keras.regularizers.l2(0.0001))(ip)
        return x


    def lr_schedule(epoch):
        '动态调整学习率'
        if epoch <= 10:
            lr = 0.1
        elif epoch <= 20:
            lr = 0.01
        elif epoch <= 30:
            lr = 0.001
        else:
            lr = 0.0001

        return lr



class SRDenseNet_H:
    def __init__(self,img_width=96, img_height=96, batch_size=16, channels=3, kernel_size=3, bolck_size=8):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.channels = channels
        self.kernel_size = kernel_size
        self.block_size = bolck_size


    def Conv_layer(self,ip):
        x = Conv2D(16,
                   kernel_size=self.kernel_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer=keras.initializers.random_normal(stddev=np.sqrt(2 / self.kernel_size/ self.kernel_size / self.img_height)),
                   bias_initializer=keras.initializers.constant(0),
                   kernel_regularizer=keras.regularizers.l2(0.0001),
                   bias_regularizer=keras.regularizers.l2(0.0001))(ip)
        return x


    def dense_block(self,ip):
        x = self.Conv_layer(ip)
        for b in range(self.block_size-1):
            y = self.Conv_layer(x)
            x = keras.layers.concatenate([x,y])
        return x

    def create_sr_model(self,ip):
        '卷积层'
        x = self.Conv_layer(ip)
        '稠密块，共8个稠密块'
        x_1 = self.dense_block(x)
        x_2 = self.dense_block(x_1)
        x_3 = self.dense_block(x_2)
        x_4 = self.dense_block(x_3)
        x_5 = self.dense_block(x_4)
        x_6 = self.dense_block(x_5)
        x_7 = self.dense_block(x_6)
        x_8 = self.dense_block(x_7)
        '反卷积层'
        y = Deconv2D(64,kernel_size=self.kernel_size,activation='relu',strides=2,padding='same')(x_8)
        y = Deconv2D(64,kernel_size=self.kernel_size,activation='relu',strides=2,padding='same')(y)
        '重构层'
        outputs = Conv2D(self.channels, kernel_size=self.kernel_size, padding='same')(y)

        model = Model(inputs=ip,outputs=outputs)
        return model



class SRDenseNet_HL:
    def __init__(self,img_width=96, img_height=96, batch_size=16, channels=3, kernel_size=3, bolck_size=8):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.channels = channels
        self.kernel_size = kernel_size
        self.block_size = bolck_size


    def Conv_layer(self,ip):
        x = Conv2D(16,
                   kernel_size=self.kernel_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer=keras.initializers.random_normal(stddev=np.sqrt(2 / self.kernel_size/ self.kernel_size / self.img_height)),
                   bias_initializer=keras.initializers.constant(0),
                   kernel_regularizer=keras.regularizers.l2(0.0001),
                   bias_regularizer=keras.regularizers.l2(0.0001))(ip)
        return x


    def dense_block(self,ip):
        x = self.Conv_layer(ip)
        for b in range(self.block_size-1):
            y = self.Conv_layer(x)
            x = keras.layers.concatenate([x,y])
        return x

    def create_sr_model(self,ip):
        '卷积层'
        x = self.Conv_layer(ip)
        '稠密块，共8个稠密块'
        x_1 = self.dense_block(x)
        x_2 = self.dense_block(x_1)
        x_3 = self.dense_block(x_2)
        x_4 = self.dense_block(x_3)
        x_5 = self.dense_block(x_4)
        x_6 = self.dense_block(x_5)
        x_7 = self.dense_block(x_6)
        x_8 = self.dense_block(x_7)
        'skip_connection'
        sk = keras.layers.concatenate([x,x_8])
        '反卷积层'
        y = Deconv2D(64,kernel_size=self.kernel_size,activation='relu',strides=2,padding='same')(sk)
        y = Deconv2D(64,kernel_size=self.kernel_size,activation='relu',strides=2,padding='same')(y)
        '重构层'
        outputs = Conv2D(self.channels, kernel_size=self.kernel_size, padding='same')(y)

        model = Model(inputs=ip,outputs=outputs)
        return model




class SRDenseNet_ALL:
    def __init__(self,img_width=96, img_height=96, batch_size=16, channels=3, kernel_size=3, bolck_size=8):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.channels = channels
        self.kernel_size = kernel_size
        self.block_size = bolck_size


    def Conv_layer(self,ip):
        x = Conv2D(16,
                   kernel_size=self.kernel_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer=keras.initializers.random_normal(stddev=np.sqrt(2 / self.kernel_size/ self.kernel_size / self.img_height)),
                   bias_initializer=keras.initializers.constant(0),
                   kernel_regularizer=keras.regularizers.l2(0.0001),
                   bias_regularizer=keras.regularizers.l2(0.0001))(ip)
        return x


    def dense_block(self,ip):
        x = self.Conv_layer(ip)
        for b in range(self.block_size-1):
            y = self.Conv_layer(x)
            x = keras.layers.concatenate([x,y])
        return x

    def create_sr_model(self,ip):
        '卷积层'
        x = self.Conv_layer(ip)
        '稠密块，共8个稠密块'
        x_1 = self.dense_block(x)
        x_2 = self.dense_block(x_1)
        x_3 = self.dense_block(x_2)
        x_4 = self.dense_block(x_3)
        x_5 = self.dense_block(x_4)
        x_6 = self.dense_block(x_5)
        x_7 = self.dense_block(x_6)
        x_8 = self.dense_block(x_7)
        'skip_connection'
        sk = keras.layers.concatenate([x,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8])
        'bottleneck'
        bn = Conv2D(128,kernel_size=1,activation='relu',padding='same')(sk)
        '反卷积层'
        y = Deconv2D(64,kernel_size=self.kernel_size,activation='relu',strides=2,padding='same')(bn)
        y = Deconv2D(64,kernel_size=self.kernel_size,activation='relu',strides=2,padding='same')(y)
        '重构层'
        outputs = Conv2D(self.channels, kernel_size=self.kernel_size, padding='same')(y)

        model = Model(inputs=ip,outputs=outputs)
        return model