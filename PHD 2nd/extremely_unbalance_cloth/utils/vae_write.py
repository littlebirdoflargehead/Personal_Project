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

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        header.insert(0, 'epoch')
        writer.writerow(header)
        epoch = torch.range(1, data.size(0)).view(data.size(0), 1)
        writer.writerows(torch.cat([epoch, data], dim=1).numpy())


def dicwrite_csv(filename, header, dic, sub_name=''):
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
        n = dic[header[0]].numel()
        data = torch.range(1, n).view(n, 1)
        for i in range(len(header)):
            data = torch.cat([data, dic[header[i]].view(n, 1)], dim=1)
        header.insert(0, 'epoch')
        writer.writerow(header)
        writer.writerows(data.numpy())
