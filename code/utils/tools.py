# coding:UTF-8
'''
各种处理 SEED Dataset 用到的小函数
Created by Xiao Guowen.
'''
import scipy.io as scio
import numpy as np
import os


def get_labels(label_path):
    '''
        得到15个 trials 对应的标签
    :param label_path: 标签文件对应的路径
    :return: list，对应15个 trials 的标签，2 for positive, 1 for neutral, 0 for negative
    '''
    return scio.loadmat(label_path, verify_compressed_data_integrity=False)['label'][0]


def get_frequency_band_idx(frequency_band):
    '''
        获得频带对应的索引，仅对 ExtractedFeatures 目录下的数据有效
    :param frequency_band: 频带名称，'delta', 'theta', 'alpha', 'beta', 'gamma'
    :return: idx，频带对应的索引
    '''
    lookup = {'delta': 0,
              'theta': 1,
              'alpha': 2,
              'beta': 3,
              'gamma': 4}
    return lookup[frequency_band]


def build_extracted_features_dataset(folder_path, feature_name, frequency_band):
    '''
        将 folder_path 文件夹中的 ExtractedFeatures 数据转化为机器学习常用的数据集
        ToDo: 增加 channel 的选择，而不是使用所有的 channel
    :param folder_path: ExtractedFeatures 文件夹对应的路径
    :param feature_name: 需要使用的特征名，如 'de_LDS'，'asm_LDS' 等，以 de_LDS1 为例，维度为 62 * 235 * 5，235为影片长度235秒，每秒切分为一个样本，62为通道数，5为频带数
    :param frequency_band: 需要选取的频带，'delta', 'theta', 'alpha', 'beta', 'gamma'
    :return: feature_vector_list, label_list. 分别为样本的特征向量，样本的标签
    '''
    frequency_idx = get_frequency_band_idx(frequency_band)
    labels = get_labels(os.path.join(folder_path, 'label.mat'))
    feature_vector_list = []
    label_list = []
    try:
        all_mat_file = os.walk(folder_path)
        skip_set = {'label.mat', 'readme.txt'}
        file_cnt = 0
        for path, dir_list, file_list in all_mat_file:
            for file_name in file_list:
                file_cnt += 1
                print('当前已处理到{}，总进度{}/{}'.format(file_name, file_cnt, len(file_list)))
                if file_name not in skip_set:
                    for trials in range(1, 16):
                        all_features_dict = scio.loadmat(os.path.join(folder_path, file_name), verify_compressed_data_integrity=False)
                        cur_feature = all_features_dict[feature_name + str(trials)]
                        cur_feature = np.asarray(cur_feature[:, :, frequency_idx]).T  # 转置后，维度为 N * 62, N 为影片长度
                        feature_vector_list.extend(_ for _ in cur_feature)
                        for sec in range(len(cur_feature)):
                            label_list.append(labels[trials - 1])
    except FileNotFoundError as e:
        print('加载数据时出错: {}'.format(e))

    return feature_vector_list, label_list


feature_vector_list, label_list = build_extracted_features_dataset('../../data/ExtractedFeatures/', 'de_LDS', 'gamma')
print(np.asarray(feature_vector_list).shape)
print(np.asarray(label_list).shape)
print(feature_vector_list[0])
print(label_list[:10])
