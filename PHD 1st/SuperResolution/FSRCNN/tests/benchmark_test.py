import keras
import keras.backend as K
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from loss import psnr, PSNRLoss, compute_ssim
from models import FSRCNN,VDSR,SRDenseNet_H,SRDenseNet_HL,SRDenseNet_ALL

import os
import time
import numpy as np
import random
from scipy.misc import imsave, imresize,imread
from scipy.ndimage import gaussian_filter



# 动态分配内存
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

base_weights_path = "weights/"
base_val_images_path = "val_images/"
base_test_images = "test_images/"

set5_path = r"/home2/liangjw/Documents/DataSet/SR/Set5/"
set14_path = r"/home2/liangjw/Documents/DataSet/SR/Set14/"
bsd100_path = r"/home2/liangjw/Documents/DataSet/SR/B100/"



scale = 4




if not os.path.exists(base_weights_path):
    os.makedirs(base_weights_path)

if not os.path.exists(base_val_images_path):
    os.makedirs(base_val_images_path)

if not os.path.exists(base_test_images):
    os.makedirs(base_test_images)


# 调用_test_loop对不同的测试集进行测试，并返回psnr值
def test_set5(model: Model, img_width=32, img_height=32, batch_size=1, normalized=True):
    datagen = ImageDataGenerator(rescale=1. / 255)
    large_img_width = img_width
    large_img_height = img_height
    img_width = int(img_width / scale)
    img_height = int(img_height / scale)

    iteration = 0
    total_psnr = 0.0

    print("Testing model on Set 5 Validation images")
    total_psnr = _test_loop(set5_path, batch_size, datagen, img_height, img_width, iteration, large_img_height,
                            large_img_width,
                            model, total_psnr, "set5", 5, normalized)

    print("Average PSNR of Set5 validation images : ", total_psnr / 5)
    print()


def test_set14(model: Model, img_width=32, img_height=32, batch_size=1, normalized=True):
    datagen = ImageDataGenerator(rescale=1. / 255)
    large_img_width = img_width
    large_img_height = img_height
    img_width = int(img_width / scale)
    img_height = int(img_height / scale)

    iteration = 0
    total_psnr = 0.0

    print("Testing model on Set 14 Validation images")
    total_psnr = _test_loop(set14_path, batch_size, datagen, img_height, img_width, iteration, large_img_height,
                            large_img_width, model, total_psnr, "set14", 14,normalized)

    print("Average PSNR of Set5 validation images : ", total_psnr / 14)
    print()


def test_bsd100(model: Model, img_width=32, img_height=32, batch_size=1, normalized=True):
    datagen = ImageDataGenerator(rescale=1. / 255)
    large_img_width = img_width
    large_img_height = img_height
    img_width = int(img_width/scale)
    img_height = int(img_height/scale)

    iteration = 0
    total_psnr = 0.0

    print("Testing model on BSD 100 Validation images")
    total_psnr = _test_loop(bsd100_path, batch_size, datagen, img_height, img_width, iteration, large_img_height,
                            large_img_width,
                            model, total_psnr, "bsd100", 100, normalized)

    print("Average PSNR of BSD100 validation images : ", total_psnr / 100)
    print()


# 使用图片自身的大小进行超分辨率处理
def test_individul(prefix='Set5',scale=4,mode='rgb',normalized=True):

    pic_path = os.path.join(set5_path[0:len(set5_path)-6], prefix)
    pic_path = os.path.join(pic_path, prefix)

    total_psnr_bicubic = 0
    total_psnr_generated = 0
    total_ssim_bicubic = 0
    total_ssim_generated = 0

    for file in os.listdir(pic_path):

        real_rgb = imread(os.path.join(pic_path, file))
        '图片数据预处理'
        img_shape = real_rgb.shape
        img_width = int(img_shape[0] / scale) * scale
        img_height = int(img_shape[1] / scale) * scale
        real_rgb = real_rgb[0:img_width,0:img_height]
        '可能出现灰度图的情况，通道只有2'
        if len(img_shape) == 2:
            real_temp = np.empty((img_width, img_height, 3))
            for l in range(3):
                real_temp[:, :, l] = real_rgb
            real_rgb = real_temp.copy()

        '对读入的RGB图像进行下取样，并对低分辨率图像进行bicubic插值'
        lr_rgb = imresize(real_rgb, 1 / scale, interp='bicubic')
        bicubic_rgb = imresize(lr_rgb, (img_width, img_height), interp='bicubic')

        if mode=='ycrcb':
            real_ycbcr = rgb2ycbcr(real_rgb)
            bicubic_ycbcr = rgb2ycbcr(bicubic_rgb)
            lr_ycbcr = rgb2ycbcr(lr_rgb)
            lr_test = np.empty((1,lr_ycbcr.shape[0],lr_ycbcr.shape[1],1))
            lr_test[0,:,:,0] = lr_ycbcr[:,:,0]
        elif mode=='rgb':
            lr_test = np.empty((1, lr_rgb.shape[0], lr_rgb.shape[1], lr_rgb.shape[2]))
            lr_test[0,:,:,:] = lr_rgb


        '读取训练后的SR模型'
        SRDenseNet_Test = SRModel(img_width=lr_test.shape[1], img_height=lr_test.shape[2], mode=mode)
        SRDenseNet_Test.build_model(load_weights=True)

        if normalized: lr_test /= 255

        sr_test = SRDenseNet_Test.model.predict(lr_test)

        if normalized: sr_test = sr_test*255

        if mode=='ycrcb':
            SR_ycrcb = np.empty((sr_test.shape[1],sr_test.shape[2],3))
            SR_ycrcb[:,:,0] = sr_test[0,:,:,0]
            SR_ycrcb[:,:,1] = real_ycbcr[:,:,1]
            SR_ycrcb[:,:,2] = real_ycbcr[:,:,2]
            SR_rgb = ycbcr2rgb(SR_ycrcb)
        elif mode=='rgb':
            SR_rgb = sr_test[0]
        SR_rgb = np.clip(SR_rgb, 0, 255)


        '保存验证集中的图片'
        real_img = np.clip(real_rgb, 0, 255).astype('uint8')
        bicubic_img = np.clip(bicubic_rgb, 0, 255).astype('uint8')
        output_image = np.clip(SR_rgb, 0, 255).astype('uint8')


        PSNR_bicubic = psnr(real_img.astype('float'),bicubic_img.astype('float'))
        PSNR_generated = psnr(real_img.astype('float'), output_image.astype('float'))

        real_ycbcr = rgb2ycbcr(real_img)
        bicubic_ycbcr = rgb2ycbcr(bicubic_img)
        output_ycbcr = rgb2ycbcr(output_image)
        SSIM_bicubic = compute_ssim(real_ycbcr[:, :, 0], bicubic_ycbcr[:, :, 0])
        SSIM_generated = compute_ssim(real_ycbcr[:, :, 0], output_ycbcr[:, :, 0])

        print('%s_PSNR/SSIM:  bicubic    %0.2f/%0.2f ; SRDenseNet   %0.2f/%0.2f' % (
            file[0:len(file) - 4], PSNR_bicubic, SSIM_bicubic, PSNR_generated, SSIM_generated))

        if not os.path.exists(base_test_images + prefix):
            os.makedirs(base_test_images + prefix)
        real_path = base_test_images + prefix + '/' + prefix + "_%s_real.png" % (file[0:len(file) - 4])
        bicubic_img_path = base_test_images + prefix + '/' + prefix + "_%s_bicubic_%0.2f|%0.2f.png" % (
            file[0:len(file) - 4], PSNR_bicubic, SSIM_bicubic)
        generated_path = base_test_images + prefix + '/' + prefix + "_%s_generated_%0.2f|%0.2f.png" % (
            file[0:len(file) - 4], PSNR_generated, SSIM_generated)

        imsave(real_path, real_img)
        imsave(bicubic_img_path, bicubic_img)
        imsave(generated_path, output_image)

        total_psnr_bicubic += PSNR_bicubic
        total_psnr_generated += PSNR_generated
        total_ssim_bicubic += SSIM_bicubic
        total_ssim_generated += SSIM_generated

    l = len(os.listdir(pic_path))
    print('Average_PSNR/SSIM:  bicubic    %0.2f/%0.2f ; FSRCNN   %0.2f/%0.2f' % (
    total_psnr_bicubic / l, total_ssim_bicubic / l, total_psnr_generated / l, total_ssim_generated / l))



# 测试调用的主函数，使用图片增强的方式进行回测
def _test_loop(path, batch_size, datagen, img_height, img_width, iteration, large_img_height, large_img_width, model,
               total_psnr, prefix, nb_images, normalized):
    """

    :param path: 数据集地址
    :param batch_size: 每个iteration生成图片数
    :param datagen: 图片增强迭代器
    :param img_height:
    :param img_width:
    :param iteration:
    :param large_img_height:
    :param large_img_width:
    :param model: 网络模型
    :param total_psnr: 总pnsr值
    :param prefix: 文件保存位置
    :param nb_images: 测试图片数量
    :param normalized: 预测模型在训练时是否将图片归一化
    :return:
    """
    for x in datagen.flow_from_directory(path, class_mode=None, batch_size=batch_size,
                                         target_size=(large_img_width, large_img_height)):
        t1 = time.time()

        # resize images
        x_temp = x.copy()
        #x_temp = x_temp.transpose((0, 2, 3, 1))

        x_generator = np.empty((batch_size, large_img_width, large_img_height, 3))

        for j in range(batch_size):

            '先对图片进行下取样，再进行bicubic插值'

            img = imresize(x_temp[j], (img_width, img_height))
            img = imresize(img, (large_img_width, large_img_height), interp='bicubic')
            '归一化情况下须要除以255'
            if normalized: x_generator[j, :, :, :] = img / 255.
            else: x_generator[j, :, :, :] = img


        output_image_batch = model.predict_on_batch(x_generator)

        average_psnr = 0.0
        for x_i in range(batch_size):
            if normalized:
                average_psnr += psnr(x[x_i], output_image_batch[x_i] )
            else:
                average_psnr += psnr(x[x_i], output_image_batch[x_i] / 255.)
            total_psnr += average_psnr

        average_psnr /= batch_size

        iteration += batch_size
        t2 = time.time()

        print("Time required : %0.2f. Average validation PSNR over %d samples = %0.2f" %
              (t2 - t1, batch_size, average_psnr))

        for x_i in range(batch_size):
            '保存验证集中的图片'
            real_path = base_test_images + prefix + "_iteration_%d_num_%d_real_.png" % (iteration, x_i + 1)
            bicubic_path = base_test_images + prefix + "_iteration_%d_num_%d_bicubic.png" % (iteration, x_i + 1)
            generated_path = base_test_images + prefix + "_iteration_%d_num_%d_generated.png" % (iteration, x_i + 1)

            val_x = x[x_i].copy() * 255.
            val_x = np.clip(val_x, 0, 255).astype('uint8')

            if normalized: input_img = x_generator[x_i].copy()*255
            else: input_img = x_generator[x_i].copy()
            input_img = np.clip(input_img, 0, 255).astype('uint8')

            if normalized: output_image = output_image_batch[x_i]*255
            else: output_image = output_image_batch[x_i]
            output_image = np.clip(output_image, 0, 255).astype('uint8')

            imsave(real_path, val_x)
            imsave(bicubic_path, input_img)
            imsave(generated_path, output_image)

        if iteration >= nb_images:
            break
    return total_psnr


mat = np.array(
    [[65.481, 128.553, 24.966],
     [-37.797, -74.203, 112.0],
     [112.0, -93.786, -18.214]])
mat_inv = np.linalg.inv(mat)
offset = np.array([16, 128, 128])


'RGB图像与YCBCR图像的转换'
def rgb2ycbcr(rgb_img):
    ycbcr_img = np.zeros(rgb_img.shape)
    ycbcr_img[:,:,0] = (mat[0,0]*rgb_img[:,:,0]+mat[0,1]*rgb_img[:,:,1]+mat[0,2]*rgb_img[:,:,2])/255+offset[0]
    ycbcr_img[:,:,1] = (mat[1,0]*rgb_img[:,:,0]+mat[1,1]*rgb_img[:,:,1]+mat[1,2]*rgb_img[:,:,2])/255+offset[1]
    ycbcr_img[:, :, 2] = (mat[2, 0] * rgb_img[:, :, 0] + mat[2, 1] * rgb_img[:, :, 1] + mat[2, 2] * rgb_img[:, :, 2])/255+offset[2]
    return ycbcr_img

def ycbcr2rgb(ycbcr_img):
    rgb_img = np.zeros(ycbcr_img.shape)
    I = np.ones((ycbcr_img.shape[0],ycbcr_img.shape[1]))
    rgb_img[:,:,0] = (mat_inv[0,0]*ycbcr_img[:,:,0]+mat_inv[0,1]*ycbcr_img[:,:,1]+mat_inv[0,2]*ycbcr_img[:,:,2]-
                      np.dot(mat_inv[0,:],offset)*I)*255
    rgb_img[:,:,1] = (mat_inv[1,0]*ycbcr_img[:,:,0]+mat_inv[1,1]*ycbcr_img[:,:,1]+mat_inv[1,2]*ycbcr_img[:,:,2]-
                      np.dot(mat_inv[1,:],offset)*I)*255
    rgb_img[:,:,2] = (mat_inv[2,0]*ycbcr_img[:,:,0]+mat_inv[2,1]*ycbcr_img[:,:,1]+mat_inv[2,2]*ycbcr_img[:,:,2]-
                      np.dot(mat_inv[2,:],offset)*I)*255
    return rgb_img


class SRModel:
    """
    自定义用于设计模型和训练模型的对象
    """
    def __init__(self, img_width=96, img_height=96, batch_size=16, mode='ycrcb'):
        assert img_width >= 16, "Minimum image width must be at least 16"
        assert img_height >= 16, "Minimum image height must be at least 16"

        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size

        self.mode = mode  # 'rgb'或'ycrcb'
        if mode=='rgb': self.channels = 3
        elif mode=='ycrcb': self.channels = 1

        self.model = None # type: Model
        self.weights_path = base_weights_path + "SRDenseNet_final_weights.h5"


    def build_model(self, load_weights=False,) -> Model:
        '创建SRDense_net网络模型对象'
        SRDense_net = FSRCNN(self.img_width, self.img_height, self.batch_size, self.channels,scale=scale)
        ip = Input(shape=(self.img_width, self.img_height, self.channels), name='x_generator')

        self.model = SRDense_net.create_sr_model(ip)
        self.model.summary()
        self.model.compile(Adam(lr=1e-4), loss='mse', metrics=[PSNRLoss])

        if load_weights:
            try:
                self.model.load_weights(self.weights_path)
                print("SRDenseNet model weights loaded.")
            except Exception:
                print("Weight for SRDenseNet model not found or are incorrect size. Cannot load weights.")

                response = input("Continue without loading weights? 'y' or 'n' ")
                if response == 'n':
                    exit()

        return self.model

    def train_model(self, image_dir, nb_images=50000, epochs=1, normalized=True):
        '使用数据增强的模型训练网络'
        datagen = ImageDataGenerator( horizontal_flip=True, vertical_flip = True) #进行水平与垂直方向翻转

        early_stop = False
        iteration = 0

        print("Training SRDenseNet")

        for i in range(epochs):
            print()
            print("Epoch : %d" % (i + 1))

            total_psnr = 0

            # current_lr = 1e-4
            # if current_lr != prev_lr:
            #     '当学习率发生变化时，改变模型的编译方式，学习率与梯度裁剪'
            #     prev_lr = current_lr
            #     # sgd = keras.optimizers.SGD(lr=current_lr, momentum=0.5, clipvalue=10*np.sqrt(0.1)/np.sqrt(current_lr))
            #     self.model.compile(Adam(lr=current_lr), loss='mse', metrics=[PSNRLoss])
            #     '适当保存模型参数'
            #     self.model.save_weights(base_weights_path + "SRDenseNet_Epoch%d_weights.h5"%(i), overwrite=True)

            for real_rgb in datagen.flow_from_directory(image_dir, class_mode=None, batch_size=self.batch_size,
                                                 target_size=(self.img_width*scale,self.img_height*scale)):
                try:
                    t1 = time.time()

                    crop_ratio = 1

                    '用于存放训练图片的变量, lr_train为低分辨率图片， sr_train为高分辨率图片'
                    lr_train = np.empty((crop_ratio*crop_ratio*self.batch_size, self.img_width, self.img_height, self.channels))
                    sr_train = np.empty((real_rgb.shape[0],real_rgb.shape[1],real_rgb.shape[2], self.channels))

                    for j in range(self.batch_size):
                        '循环对每张图进行下取样处理，最终生成训练集'
                        '对图像进行下取样'
                        lr_rgb = imresize(real_rgb[j], size=1 / scale, interp='bicubic')
                        if self.mode=='ycrcb':
                            real_ycrcb = rgb2ycbcr(real_rgb[j])
                            lr_ycrcb = rgb2ycbcr(lr_rgb)
                            '只取ycbcr中的y通道进行训练'
                            lr_train[j,:,:,0] = lr_ycrcb[:,:,0]
                            sr_train[j,:,:,0] = real_ycrcb[:,:,0]
                        elif self.mode=='rgb':
                            lr_train[j,:,:,:] = lr_rgb
                            sr_train[j,:,:,:] = real_rgb[j]
                            '保存临时图片'
                            # img_temp = np.clip(lr_rgb, 0, 255).astype('uint8')
                            # x_real_temp = np.clip(real_ycrcb[j], 0, 255).astype('uint8')
                            #
                            # x_real_path = base_val_images_path + \
                            #               "epoch_%d_iteration_%d_num_%d_read.png" % (
                            #                   i + 1, iteration, j + 1)
                            # downsampling_path = base_val_images_path + \
                            #                     "epoch_%d_iteration_%d_num_%d_generated.png" % (
                            #                         i + 1, iteration, j + 1)
                            # imsave(x_real_path, x_real_temp)
                            # imsave(downsampling_path, img_temp)

                    if normalized:
                        lr_train /=255
                        sr_train /=255

                    hist = self.model.fit(lr_train, sr_train, batch_size=self.batch_size*crop_ratio*crop_ratio, epochs=1, verbose=0)
                    psnr_loss_val = hist.history['PSNRLoss'][0]
                    if normalized: psnr_loss_val -= 48.1308036087

                    total_psnr += psnr_loss_val*self.batch_size

                    iteration += self.batch_size
                    t2 = time.time()

                    print("Iter : %d / %d | Time required : %0.2f seconds | "
                          "PSNR : %0.3f" % (iteration, nb_images, t2 - t1, psnr_loss_val))

                    if iteration % 1000 == 0 and iteration != 0:
                        print("Saving weights")
                        self.model.save_weights(self.weights_path, overwrite=True)

                    if iteration >= nb_images:
                        break

                except KeyboardInterrupt:
                    print("Keyboard interrupt detected. Stopping early.")
                    early_stop = True
                    break

            iteration = 0

            self.model.save_weights(base_weights_path + "SRDenseNet_Epoch%d_weights%0.2f.h5" % (9, total_psnr/nb_images), overwrite=True)

            if early_stop:
                    break

        print("Finished training VDSR network. Saving model weights.")
        self.model.save_weights(self.weights_path, overwrite=True)








train_path = r"/home2/liangjw/Documents/DataSet/coco2014"

# 该参数为训练时网络的输入size，必须整除256
# img_width = img_height = 100
#
# SRDenseNet_Test = SRModel(img_width=img_width, img_height=img_height, batch_size=20, mode = 'ycrcb')
# SRDenseNet_Test.build_model(load_weights=True)
# SRDenseNet_Test.train_model(train_path, nb_images=50000, epochs=1, normalized=True)


# 预测时使用更大的输入层(对每一张图片缩放成相同大小的图片)
# img_width = img_height = 256
# VDSR_Test = SRModel(img_width=img_width, img_height=img_height, batch_size=3)
# VDSR_Test.build_model(load_weights=True)
#
# test_set5(VDSR_Test.model, img_width=img_width, img_height=img_height, normalized=False)
# test_set14(VDSR_Test.model, img_width=img_width, img_height=img_height, normalized=False)
# test_bsd100(VDSR_Test.model, img_width=img_width, img_height=img_height, normalized=False)


# 预测时使用更大的输入层(使用原图)
test_individul(prefix='Set5',scale=4, mode = 'ycrcb', normalized=True)
test_individul(prefix='Set14',scale=4,mode = 'ycrcb', normalized=True)