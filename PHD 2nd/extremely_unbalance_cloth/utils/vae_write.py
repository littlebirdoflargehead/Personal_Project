import torch
import numpy as np


def write_csv(filename, header, data, sub_name=''):
    '''
    将数据接入csv中
    :param filename: csv文件名
    :param header: 标题
    :param data: 保存数据的字典（须要以tensor的形式保存）
    :param sub_name: 辅文件名
    '''
    import csv
    if '.' in filename:
        s = filename.split('.')
        filename = s[0] + '_' + sub_name + '.' + s[1]
    else:
        filename = filename + '_' + sub_name + '.csv'

    if type(data) is np.ndarray:
        data = torch.from_numpy(data)

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        header.insert(0, 'Epoch')
        writer.writerow(header)
        epoch = torch.arange(0, data.size(0)).float().view(data.size(0), 1) + 1
        writer.writerows(torch.cat([epoch, data.float()], dim=1).numpy())


def dic_write_csv(filename, header, dic, sub_name=''):
    '''
    将字典数据接入csv中
    :param filename: csv文件名
    :param header: 标题
    :param dic: 字典数据
    :param sub_name: 辅文件名
    '''
    import csv
    if '.' in filename:
        s = filename.split('.')
        filename = s[0] + '_' + sub_name + '.' + s[1]
    else:
        filename = filename + '_' + sub_name + '.csv'

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        n = dic[header[0]].shape[0]
        data = torch.arange(0, n).float().view(n, 1) + 1
        for i in range(len(header)):
            if type(dic[header[i]]) is np.ndarray:
                dic[header[i]] = torch.from_numpy(dic[header[i]])
            data = torch.cat([data, dic[header[i]].float().view(n, 1)], dim=1)
        header.insert(0, 'Epoch')
        writer.writerow(header)
        writer.writerows(data.numpy())


def make_threshold_titles(threshold,titles):
    '''
    根据阈值以及标题生成卷积后的标题
    :param threshold: 阈值
    :param titles: 各种标题
    :return: 最终生成的卷积标题列表
    '''
    Titles = []
    for i in range(len(threshold)):
        for j in range(len(titles)):
            Titles.append(titles[j]+'_{}'.format(threshold[i]))
    return Titles
