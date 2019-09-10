import torch
import numpy as np
from sklearn import preprocessing
import time
from ModelAndProcess import pinv,iter_pinv,ada_weight,FeatureMap,EnhancementNode,show_accuracy,AveWeight,show_accuracy_vote



def BLS(train_x,train_y,test_x,test_y,s,c,N1,N2,N3):
    # 将数据的每一输入维度都进行标准化，均值为0，方差为1，并保存为tensor的形式
    train_x = torch.from_numpy(preprocessing.scale(train_x,axis = 1)).double()
    test_x = torch.from_numpy(preprocessing.scale(test_x,axis = 1)).double()

    # 训练阶段
    time_start = time.time()

    # FeatureMaps层，FeatureMaps为tensor形式的特征层的输出N*(N1*N2)，FeatureLayers为列表格式保留各个特征网络中的参数
    FeatureLayers = []
    for i in range(N2):
        featurelayer = FeatureMap(input_dim=train_x.shape[1],output_dim=N1)
        FeatureLayers.append(featurelayer)
        featuremap = featurelayer(train_x,train=True)
        if i==0:
            FeatureMaps = featuremap
        else:
            FeatureMaps = torch.cat([FeatureMaps,featuremap],dim=1)

    # EnhancementNode层，EnhancementNodes为tensor形式的增强层输出N*N3，enhancementlayer为pytorch中的网络形式
    enhancementlayer = EnhancementNode(input_dim=N1*N2,output_dim=N3)
    EnhancementNodes = enhancementlayer(FeatureMaps)

    # 使用矩阵伪逆求出对应的参数矩阵W
    A = torch.cat([FeatureMaps,EnhancementNodes],dim=1)
    A_inv = pinv(A,c)
    W = torch.mm(A_inv,train_y)

    time_end = time.time()

    # 在训练集上统计训练时间和训练准确率
    train_out = torch.mm(A,W)
    trainTime = time_end-time_start
    trainAcc = show_accuracy(train_out, train_y)
    print('Training accurate is %5.3f ' % (trainAcc*100) ,'%')
    print('Training time is %5.3f s' %trainTime)

    # 测试阶段
    time_start = time.time()

    # FeatureMaps层，读取FeatureLayers列表中的网络结构和参数，并返回Test_FeatureMaps
    for i in range(len(FeatureLayers)):
        featurelayer = FeatureLayers[i]
        featuremap = featurelayer(test_x)
        if i==0:
            test_FeatureMaps = featuremap
        else:
            test_FeatureMaps = torch.cat([test_FeatureMaps,featuremap],dim=1)

    # EnhancementNode层，使用enhancementlayer输出结果
    test_EnhancementNodes = enhancementlayer(test_FeatureMaps)

    # 使用W求出测试结果
    test_A = torch.cat([test_FeatureMaps,test_EnhancementNodes],dim=1)
    test_out = torch.mm(test_A,W)

    time_end = time.time()

    # 在测试集上统计测试时间和测试准确率
    testTime = time_end - time_start
    testAcc = show_accuracy(test_out, test_y)
    print('Testing accurate is %5.3f ' % (testAcc*100) ,'%')
    print('Testing time is %5.3f s' %testTime)

    return testAcc,testTime,trainAcc,trainTime



def BLS_AddEnhanceNodes(train_x,train_y,test_x,test_y,s,c,N1,N2,N3,L,M):
    # 保存实验所需要的数据
    train_acc = np.zeros([1, L + 1])
    test_acc = np.zeros([1, L + 1])
    train_time = np.zeros([1, L + 1])
    test_time = np.zeros([1, L + 1])

    # 将数据的每一输入维度都进行标准化，均值为0，方差为1，并保存为tensor的形式
    train_x = torch.from_numpy(preprocessing.scale(train_x, axis=1)).double()
    test_x = torch.from_numpy(preprocessing.scale(test_x, axis=1)).double()

    '''
    构建基础网络
    '''
    # 训练阶段
    time_start = time.time()

    # FeatureMaps层，FeatureMaps为tensor形式的特征层的输出N*(N1*N2)，FeatureLayers为列表格式保留各个特征网络中的参数
    FeatureLayers = []
    for i in range(N2):
        featurelayer = FeatureMap(input_dim=train_x.shape[1], output_dim=N1)
        FeatureLayers.append(featurelayer)
        featuremap = featurelayer(train_x, train=True)
        if i == 0:
            FeatureMaps = featuremap
        else:
            FeatureMaps = torch.cat([FeatureMaps, featuremap], dim=1)

    # EnhancementNode层，EnhancementNodes为tensor形式的增强层输出N*N3，enhancementlayer为pytorch中的网络形式
    enhancementlayer = EnhancementNode(input_dim=N1 * N2, output_dim=N3)
    EnhancementNodes = enhancementlayer(FeatureMaps)

    # 使用矩阵伪逆求出对应的参数矩阵W
    A = torch.cat([FeatureMaps, EnhancementNodes], dim=1)
    A_inv = pinv(A, 2**-25)
    W = torch.mm(A_inv, train_y)

    time_end = time.time()
    trainTime = time_end - time_start

    # 在训练集上统计训练时间和训练准确率
    train_out = torch.mm(A, W)
    trainAcc = show_accuracy(train_out, train_y)
    train_acc[0][0] = trainAcc
    train_time[0][0] = trainTime
    print('Training accurate is %5.3f ' % (trainAcc*100) ,'%')
    print('Training time is %5.3f s' %trainTime)

    # 测试阶段
    time_start = time.time()

    # FeatureMaps层，读取FeatureLayers列表中的网络结构和参数，并返回Test_FeatureMaps
    for i in range(len(FeatureLayers)):
        featurelayer = FeatureLayers[i]
        featuremap = featurelayer(test_x)
        if i == 0:
            test_FeatureMaps = featuremap
        else:
            test_FeatureMaps = torch.cat([test_FeatureMaps, featuremap], dim=1)

    # EnhancementNode层，使用enhancementlayer输出结果
    test_EnhancementNodes = enhancementlayer(test_FeatureMaps)

    # 使用W求出测试结果
    test_A = torch.cat([test_FeatureMaps, test_EnhancementNodes], dim=1)
    test_out = torch.mm(test_A, W)

    time_end = time.time()
    testTime = time_end - time_start

    # 在测试集上统计测试时间和测试准确率
    testAcc = show_accuracy(test_out, test_y)
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime
    print('Testing accurate is %5.3f ' % (testAcc*100) ,'%')
    print('Testing time is %5.3f s' %testTime)

    '''
    Incremental Enhancement Node 增量学习加入增强化节点
    '''
    # EnhancementNodes层，EnhancementNodes为tensor形式的增强节点输出N*(N3+L*M)，EnhancementLayers为列表格式保留各个增强节点网络中的参数
    EnhancementLayers = [enhancementlayer]
    for j in range(L):
        print('Incremental Enhancement Node增量学习阶段 %d' %(j+1))
        # 训练阶段
        time_start = time.time()

        enhancementlayer = EnhancementNode(input_dim=N1 * N2, output_dim=M)
        EnhancementLayers.append(enhancementlayer)
        enhancementnode = enhancementlayer(FeatureMaps)

        # 使用增量计算公式迭代计算矩阵伪逆()
        # 注意：此处使用迭代公式计算伪逆会出现一定的数值误差，与直接计算的结果有一定偏离，偏离微小，但会出现A_inv*A!=I
        # 并且由于不断迭代的关系，真实的伪逆与迭代计算得到的伪逆的差距会越来越大
        A,A_inv = iter_pinv(A,A_inv,enhancementnode,0)
        # A = torch.cat([A,enhancementnode],dim=1)
        # A_inv = pinv(A,c)
        W = torch.mm(A_inv, train_y)

        time_end = time.time()
        trainTime = time_end - time_start

        # 在训练集上统计训练时间和训练准确率
        train_out = torch.mm(A, W)
        trainAcc = show_accuracy(train_out, train_y)
        train_acc[0][j+1] = trainAcc
        train_time[0][j+1] = trainTime
        print('Training accurate is %5.3f ' % (trainAcc*100) ,'%')
        print('Training time is %5.3f s' % trainTime)

        # 测试阶段
        time_start = time.time()

        # 计算增量EnhancementNodes在测试集上的输出
        test_Incremental_EnhancementNodes = enhancementlayer(test_FeatureMaps)

        # 使用W求出测试结果
        test_A = torch.cat([test_A, test_Incremental_EnhancementNodes], dim=1)
        test_out = torch.mm(test_A, W)

        time_end = time.time()
        testTime = time_end - time_start

        # 在测试集上统计测试时间和测试准确率
        testAcc = show_accuracy(test_out, test_y)
        test_acc[0][j+1] = testAcc
        test_time[0][j+1] = testTime
        print('Testing accurate is %5.3f ' % (testAcc*100) ,'%')
        print('Testing time is %5.3f s' % testTime)

    return testAcc, testTime, trainAcc, trainTime



def BLS_Bagging(train_x,train_y,test_x,test_y,s,c,N1,N2,N3,N,Ensemble='ave',avemethod='exp'):
    # 将数据的每一输入维度都进行标准化，均值为0，方差为1，并保存为tensor的形式
    train_x = torch.from_numpy(preprocessing.scale(train_x, axis=1)).double()
    test_x = torch.from_numpy(preprocessing.scale(test_x, axis=1)).double()

    # 训练阶段
    time_start = time.time()

    # FeatureMaps层，FeatureMaps为tensor形式的特征层的输出N*(N1*N2)，FeatureLayers为列表格式保留各个特征网络中的参数
    FeatureLayers = []
    for i in range(N2):
        featurelayer = FeatureMap(input_dim=train_x.shape[1], output_dim=N1)
        FeatureLayers.append(featurelayer)
        featuremap = featurelayer(train_x, train=True)
        if i == 0:
            FeatureMaps = featuremap
        else:
            FeatureMaps = torch.cat([FeatureMaps, featuremap], dim=1)

    # EnhancementNodes层，EnhancementLayers为列表格式保留各个增强节点网络中的参数，EnhancementWeights保存每一个EnhancementNode中的权重
    EnhancementLayers = []
    EnhancementWeights = []
    EnhancementAccuracy = np.empty(N)
    for j in range(N):
        enhancementlayer = EnhancementNode(input_dim=N1*N2, output_dim=N3)
        EnhancementLayers.append(enhancementlayer)
        enhancementnode = enhancementlayer(FeatureMaps)

        # 对每一个enhancementnode计算出对应的权重，并保存在列表EnhancementWeights中
        A = torch.cat([FeatureMaps,enhancementnode],dim=1)
        A_inv = pinv(A,c)
        W = torch.mm(A_inv,train_y)
        EnhancementWeights.append(W)

        if Ensemble=='ave':
            # 当使用平均的集成方式时，统计每一个enhancementnode在训练集上的正确率，用于确定集成权重
            # 当使用其他的集成方式时，无须统计正确率
            train_out = torch.mm(A, W)
            trainAcc = show_accuracy(train_out, train_y)
            EnhancementAccuracy[j] = trainAcc
            if j==N-1:
                EnsembleWeights = AveWeight(EnhancementAccuracy,method=avemethod)

    time_end = time.time()

    # 在训练集上统计训练时间和训练准确率
    trainTime = time_end - time_start
    '''
    集成阶段
    '''
    train_out = torch.zeros(train_y.shape)
    for i in range(N):
        enhancementnode = EnhancementLayers[i](FeatureMaps)
        A = torch.cat([FeatureMaps,enhancementnode],dim=1)
        # 判断使用何种集成方式
        if Ensemble == 'ave':
            train_out = train_out+torch.mm(A,EnhancementWeights[i])*EnsembleWeights[i]
            if i==N-1:
                trainAcc = show_accuracy(train_out, train_y)
        if Ensemble == 'vote':
            if i==0:
                train_out = torch.argmax(torch.mm(A,EnhancementWeights[i]),dim=1).view(-1,1)
            else:
                train_out = torch.cat([train_out, torch.argmax(torch.mm(A, EnhancementWeights[i]), dim=1).view(-1, 1)],dim=1)
            if i==N-1 :
                trainAcc = show_accuracy_vote(train_out, train_y)

    print('Training accurate is %5.3f ' % (trainAcc * 100),'%')
    print('Training time is %5.3f s' % trainTime)

    # 测试阶段
    time_start = time.time()

    # FeatureMaps层，读取FeatureLayers列表中的网络结构和参数，并返回Test_FeatureMaps
    for i in range(len(FeatureLayers)):
        featurelayer = FeatureLayers[i]
        featuremap = featurelayer(test_x)
        if i == 0:
            test_FeatureMaps = featuremap
        else:
            test_FeatureMaps = torch.cat([test_FeatureMaps, featuremap], dim=1)

    # EnhancementNodes层，EnhancementLayers为列表格式保留各个增强节点网络中的参数，EnhancementWeights保存每一个EnhancementNode中的权重
    test_out = torch.zeros(test_y.shape)
    for j in range(len(EnhancementLayers)):
        enhancementnode = EnhancementLayers[j](test_FeatureMaps)
        test_A = torch.cat([test_FeatureMaps, enhancementnode], dim=1)
        # 判断使用何种集成方式
        if Ensemble == 'ave':
            test_out = test_out + torch.mm(test_A, EnhancementWeights[j]) * EnsembleWeights[j]
            if j == N - 1:
                testAcc = show_accuracy(test_out, test_y)
        if Ensemble == 'vote':
            if j == 0:
                test_out = torch.argmax(torch.mm(test_A, EnhancementWeights[j]), dim=1).view(-1, 1)
            else:
                test_out = torch.cat([test_out,torch.argmax(torch.mm(test_A, EnhancementWeights[j]), dim=1).view(-1, 1)], dim=1)
            if j == N-1:
                testAcc = show_accuracy_vote(test_out, test_y)


    time_end = time.time()

    # 在测试集上统计测试时间和测试准确率
    testTime = time_end - time_start
    print('Testing accurate is %5.3f ' % (testAcc * 100), '%')
    print('Testing time is %5.3f s' % testTime)

    return testAcc, testTime, trainAcc, trainTime



def BLS_AdaBoost(train_x,train_y,test_x,test_y,s,c,N1,N2,N3,N,Ensemble='ave',avemethod='exp'):
    # 将数据的每一输入维度都进行标准化，均值为0，方差为1，并保存为tensor的形式
    train_x = torch.from_numpy(preprocessing.scale(train_x, axis=1)).double()
    test_x = torch.from_numpy(preprocessing.scale(test_x, axis=1)).double()

    # 训练阶段
    time_start = time.time()

    # FeatureMaps层，FeatureMaps为tensor形式的特征层的输出N*(N1*N2)，FeatureLayers为列表格式保留各个特征网络中的参数
    FeatureLayers = []
    for i in range(N2):
        featurelayer = FeatureMap(input_dim=train_x.shape[1], output_dim=N1)
        FeatureLayers.append(featurelayer)
        featuremap = featurelayer(train_x, train=True)
        if i == 0:
            FeatureMaps = featuremap
        else:
            FeatureMaps = torch.cat([FeatureMaps, featuremap], dim=1)

    # EnhancementNodes层，EnhancementLayers为列表格式保留各个增强节点网络中的参数，EnhancementWeights保存每一个EnhancementNode中的权重
    # SampleWeight保存每个样本在AdaBoost中的权重
    EnhancementLayers = []
    EnhancementWeights = []
    EnhancementAccuracy = np.empty(N)
    SampleWeight = np.ones(train_y.shape[0])
    for j in range(N):
        enhancementlayer = EnhancementNode(input_dim=N1 * N2, output_dim=N3)
        EnhancementLayers.append(enhancementlayer)
        enhancementnode = enhancementlayer(FeatureMaps)

        # 对每一个enhancementnode计算出对应的权重，并保存在列表EnhancementWeights中
        A = torch.cat([FeatureMaps, enhancementnode], dim=1)
        W = ada_weight(A,train_y,SampleWeight,c)
        EnhancementWeights.append(W)

        # 统计正确率和每个样本是否被正确分类，增加没有被正确分类的样本的权重
        count = 0
        label = torch.argmax(train_y, dim=1)
        predlabe = torch.argmax(A.mm(W), dim=1)
        for n in range(label.shape[0]):
            if label[n] == predlabe[n]:
                count += 1
            else:
                SampleWeight[n] = SampleWeight[n]+25
        EnhancementAccuracy[j] = count/label.shape[0]

        if Ensemble == 'ave' and j==N-1:
            EnsembleWeights = AveWeight(EnhancementAccuracy, method=avemethod)

    time_end = time.time()

    # 在训练集上统计训练时间和训练准确率
    trainTime = time_end - time_start

    '''
    集成阶段
    '''
    train_out = torch.zeros(train_y.shape)
    for i in range(N):
        enhancementnode = EnhancementLayers[i](FeatureMaps)
        A = torch.cat([FeatureMaps,enhancementnode],dim=1)
        # 判断使用何种集成方式
        if Ensemble == 'ave':
            train_out = train_out+torch.mm(A,EnhancementWeights[i])*EnsembleWeights[i]
            if i==N-1:
                trainAcc = show_accuracy(train_out, train_y)
        if Ensemble == 'vote':
            if i==0:
                train_out = torch.argmax(torch.mm(A,EnhancementWeights[i]),dim=1).view(-1,1)
            else:
                train_out = torch.cat([train_out, torch.argmax(torch.mm(A, EnhancementWeights[i]), dim=1).view(-1, 1)],dim=1)
            if i==N-1 :
                trainAcc = show_accuracy_vote(train_out, train_y)

    print('Training accurate is %5.3f ' % (trainAcc * 100),'%')
    print('Training time is %5.3f s' % trainTime)

    # 测试阶段
    time_start = time.time()

    # FeatureMaps层，读取FeatureLayers列表中的网络结构和参数，并返回Test_FeatureMaps
    for i in range(len(FeatureLayers)):
        featurelayer = FeatureLayers[i]
        featuremap = featurelayer(test_x)
        if i == 0:
            test_FeatureMaps = featuremap
        else:
            test_FeatureMaps = torch.cat([test_FeatureMaps, featuremap], dim=1)

    # EnhancementNodes层，EnhancementLayers为列表格式保留各个增强节点网络中的参数，EnhancementWeights保存每一个EnhancementNode中的权重
    test_out = torch.zeros(test_y.shape)
    for j in range(len(EnhancementLayers)):
        enhancementnode = EnhancementLayers[j](test_FeatureMaps)
        test_A = torch.cat([test_FeatureMaps, enhancementnode], dim=1)
        # 判断使用何种集成方式
        if Ensemble == 'ave':
            test_out = test_out + torch.mm(test_A, EnhancementWeights[j]) * EnsembleWeights[j]
            if j == N - 1:
                testAcc = show_accuracy(test_out, test_y)
        if Ensemble == 'vote':
            if j == 0:
                test_out = torch.argmax(torch.mm(test_A, EnhancementWeights[j]), dim=1).view(-1, 1)
            else:
                test_out = torch.cat(
                    [test_out, torch.argmax(torch.mm(test_A, EnhancementWeights[j]), dim=1).view(-1, 1)], dim=1)
            if j == N - 1:
                testAcc = show_accuracy_vote(test_out, test_y)

    time_end = time.time()

    # 在测试集上统计测试时间和测试准确率
    testTime = time_end - time_start
    print('Testing accurate is %5.3f ' % (testAcc * 100), '%')
    print('Testing time is %5.3f s' % testTime)

    return testAcc, testTime, trainAcc, trainTime