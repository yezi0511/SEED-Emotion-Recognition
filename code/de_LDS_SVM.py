# coding:UTF-8
'''
使用提取的 de_LDS 特征进行情感分类，分类器使用 SVM，快速验证。
Created by Xiao Guowen.
'''
from utils.tools import build_extracted_features_dataset
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix


# 样本加载
folder_path = '../data/ExtractedFeatures/'
de_LDS_feature_list, de_LDS_label_list = build_extracted_features_dataset(folder_path, 'de_LDS', 'gamma')
de_LDS_feature_list = de_LDS_feature_list
de_LDS_label_list = de_LDS_label_list

# 训练集，测试集分割
train_feature, test_feature, train_label, test_label = train_test_split(de_LDS_feature_list, de_LDS_label_list, test_size=0.20)

# SVM 分类器训练与预测
# 注意 SVC 参数设置，默认的 C 值为1，即不容错，此时搭配 linear 核很可能无法收敛。
svc_classifier = svm.SVC(C=0.6, kernel='rbf')
svc_classifier.fit(train_feature, train_label)
pred_label = svc_classifier.predict(test_feature)
print(confusion_matrix(test_label, pred_label))
print(classification_report(test_label, pred_label))
