# coding:UTF-8
'''
对原始的 eeg 信号，使用 CNN 进行情感分类。
Created by Xiao Guowen.
'''
from utils.tools import build_preprocessed_eeg_dataset_CNN, RawEEGDataset, subject_independent_data_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# 加载数据，整理成所需要的格式
folder_path = '../data/Preprocessed_EEG/'
feature_vector_dict, label_dict = build_preprocessed_eeg_dataset_CNN(folder_path)
train_feature, train_label, test_feature, test_label = subject_independent_data_split(feature_vector_dict, label_dict,
                                                                                      {'2', '6', '9'})

desire_shape = [1, 62, 200]
train_data = RawEEGDataset(train_feature, train_label, desire_shape)
test_data = RawEEGDataset(test_feature, test_label, desire_shape)

# 超参数设置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 30
num_classes = 3
batch_size = 24
learning_rate = 0.0001

# Data loader
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)


# 定义卷积网络结构
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(128 * 7 * 25, 256, bias=True)
        self.fc2 = nn.Linear(256, num_classes, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)


# Train the model
def train():
    writer = SummaryWriter('../log')
    total_step = len(train_data_loader)
    batch_cnt = 0
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_data_loader):
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            batch_cnt += 1
            writer.add_scalar('train_loss', loss, batch_cnt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1,
                                                                         total_step, loss.item()))
        scheduler.step()
        test()
    torch.save(model.state_dict(), '../model/model.ckpt')


# Test the model
def test(is_load=False):
    if is_load:
        model.load_state_dict(torch.load('../model/model.ckpt'))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in test_data_loader:
            features = features.to(device)
            labels = labels.to(device)
            output = model(features)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy is {}%'.format(100 * correct / total))


train()
test(is_load=True)
