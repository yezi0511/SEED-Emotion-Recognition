# coding:UTF-8
'''
使用提取的 de_LDS 特征进行情感分类，分类器使用 SVM，快速验证。
Created by Xiao Guowen.
'''
from utils.tools import build_extracted_features_dataset
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


folder_path = '../data/ExtractedFeatures/'


def all_data_svm(folder_path):
    '''
        加载所有被试的数据，按一定比例切分数据集，用已提取的特征进行分类，不考虑 subject-independent / subject-dependent，也不考虑被试间的差异，粗暴求平均
    :param folder_path: ExtractedFeatures 文件夹路径
    :return one:
    '''
    # 样本加载
    de_LDS_feature_dict, de_LDS_label_dict = build_extracted_features_dataset(folder_path, 'de_LDS', 'gamma')
    de_LDS_feature_list = []
    de_LDS_label_list = []
    for key in de_LDS_feature_dict.keys():
        de_LDS_feature_list.extend(de_LDS_feature_dict[key])
        de_LDS_label_list.extend(de_LDS_label_dict[key])

    # 训练集，测试集分割
    test_ratio = 0.4
    train_feature, test_feature, train_label, test_label = train_test_split(de_LDS_feature_list, de_LDS_label_list,
                                                                            test_size=test_ratio)

    # SVM 分类器训练与预测
    # 注意 SVC 参数设置，默认的 C 值为1，即不容错，此时搭配 linear 核很可能无法收敛。
    svc_classifier = svm.SVC(C=0.7, kernel='linear')
    svc_classifier.fit(train_feature, train_label)
    pred_label = svc_classifier.predict(test_feature)
    print(confusion_matrix(test_label, pred_label))
    print(classification_report(test_label, pred_label))


def paper_svm(folder_path):
    '''
        按照 SEED 数据集原始论文中的 SVM 计算方式测试准确率和方差，每个 experiment 分开计算，取其中 9 个 trial 为训练集，6 个 trial 为测试集
    :param folder_path: ExtractedFeatures 文件夹路径
    :return None:
    '''
    # 样本加载
    de_LDS_feature_dict, de_LDS_label_dict = build_extracted_features_dataset(folder_path, 'de_LDS', 'gamma')
    accuracy = 0
    for key in de_LDS_feature_dict.keys():
        print('当前处理到 experiment_{}'.format(key))
        cur_feature = de_LDS_feature_dict[key]
        cur_label = de_LDS_label_dict[key]
        train_feature = []
        train_label = []
        test_feature = []
        test_label = []
        for trial in cur_feature.keys():
            if int(trial) < 10:
                train_feature.extend(cur_feature[trial])
                train_label.extend(cur_label[trial])
            else:
                test_feature.extend(cur_feature[trial])
                test_label.extend(cur_label[trial])
        # 定义 svm 分类器
        svc_classifier = svm.SVC(C=0.8, kernel='rbf')
        svc_classifier.fit(train_feature, train_label)
        pred_label = svc_classifier.predict(test_feature)
        print(confusion_matrix(test_label, pred_label))
        print(classification_report(test_label, pred_label))
        cur_accuracy = svc_classifier.score(test_feature, test_label)
        accuracy += cur_accuracy
        print('当前 experiment 的 accuracy 为：{}'.format(cur_accuracy))

    print('所有 experiment 上的平均 accuracy 为：{}'.format(accuracy / len(de_LDS_feature_dict.keys())))


paper_svm(folder_path)
