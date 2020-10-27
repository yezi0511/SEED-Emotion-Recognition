# coding:UTF-8
'''
使用提取的 de_LDS 特征进行情感分类，分类器使用 SVM，快速验证。
Created by Xiao Guowen.
'''
from utils


import scipy.io as scio
import numpy as np
import os


extracted_feature_path = '../data/ExtractedFeatures/'
labels = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]  # 2 for positive, 1 for neutral, 0 for negative
de_LDS_data = []
skip_set = {'label.mat', 'readme.txt'}
try:
    all_mat_file = os.walk(extracted_feature_path)
    for path, dir_list, file_list in all_mat_file:
        for file_name in file_list:
            if file_name not in skip_set:
                de_LDS_data.append((scio.loadmat(os.path.join(extracted_feature_path, file_name),
                                                 verify_compressed_data_integrity=False)[
                    'de_LDS15']))
                '''
                for video_clips in range(15):
                    de_LDS_data.append((scio.loadmat(os.path.join(extracted_feature_path, file_name), verify_compressed_data_integrity=False)['de_LDS{}'.format(video_clips + 1)]))
                '''
except FileNotFoundError as e:
    print('加载数据时出错: {}'.format(e))

print(np.array(de_LDS_data).shape)
