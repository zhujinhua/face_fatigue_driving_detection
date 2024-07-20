# 引入PyTorch
import torch
from torch import nn


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()
        # 特征抽取
        self.feature_extractor = nn.Sequential(
            # 第1层卷积
            nn.Conv2d(in_channels=3,
                      out_channels=10,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(num_features=10),
            nn.PReLU(num_parameters=10, init=0.25),

            # 最大池化
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # 第2层卷积
            nn.Conv2d(in_channels=10,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU(num_parameters=16, init=0.25),

            # 第3层卷积
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU(num_parameters=32, init=0.25)
        )
        # [N, 32, 1, 1] --> [N, 1, 1, 1]
        # 输出人脸的概率 bce 输出信息编码在了通道这个维度上
        self.conv4_1 = nn.Conv2d(in_channels=32,
                                 out_channels=1,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        # [N, 32, 1, 1] --> [N, 4, 1, 1]
        # 输出人脸的定位框的偏移量（误差）
        self.conv4_2 = nn.Conv2d(in_channels=32,
                                 out_channels=4,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

    def forward(self, x):
        # 抽取特征 [N, 32, 1, 1]
        x = self.feature_extractor(x)
        # [N, 32, 1, 1] --> [N, 1, 1, 1]
        probs = torch.sigmoid(self.conv4_1(x))
        # [N, 32, 1, 1] --> [N, 4, 1, 1]
        offset = self.conv4_2(x)
        return probs, offset


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.feature_extractor = nn.Sequential(

            # 第一层卷积
            nn.Conv2d(in_channels=3,
                      out_channels=28,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(num_features=28),
            nn.PReLU(num_parameters=28, init=0.25),

            # 第1个池化层
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1),

            # 第2层卷积
            nn.Conv2d(in_channels=28,
                      out_channels=48,
                      kernel_size=3,
                      stride=1,
                      padding=0),

            nn.BatchNorm2d(num_features=48),
            nn.PReLU(num_parameters=48, init=0.25),

            # 第2个池化层
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=0),

            # 第3层卷积
            nn.Conv2d(in_channels=48,
                      out_channels=64,
                      kernel_size=2,
                      stride=1,
                      padding=0),

            nn.BatchNorm2d(num_features=64),
            nn.PReLU(num_parameters=64, init=0.25),
        )

        self.linear4 = nn.Sequential(
            nn.Linear(in_features=64 * 3 * 3, out_features=128),
            nn.PReLU(128)
        )

        # 类别
        self.linear5_1 = nn.Linear(in_features=128, out_features=1)
        # 误差回归
        self.linear5_2 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.linear4(x)
        # 概率
        probs = torch.sigmoid(self.linear5_1(x))
        # 误差
        offset = self.linear5_2(x)
        return probs, offset


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.pre_layer = nn.Sequential(
            # 第1层卷积
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=3,
                      stride=1),  # 46
            nn.BatchNorm2d(num_features=32),
            nn.PReLU(num_parameters=32, init=0.25),

            # 第1层池化
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1),  # 23

            # 第2层卷积
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=0),  # 21
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(num_parameters=64, init=0.25),

            # 第2层池化
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=0),  # 10

            # 第3层卷积
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=0),  # 8
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(num_parameters=64, init=0.25),

            # 第3个池化层
            nn.MaxPool2d(kernel_size=2,
                         stride=2,
                         padding=0),  # 4

            # 第4层卷积
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=2,
                      stride=1,
                      padding=0),  # 3
            nn.BatchNorm2d(num_features=128),
            nn.PReLU(num_parameters=128, init=0.25)
        )

        self.linear5 = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.PReLU(num_parameters=256, init=0.25)
        )

        # 类别
        self.linear6_1 = nn.Linear(in_features=256, out_features=1)
        # 4个误差回归
        self.linear6_2 = nn.Linear(in_features=256, out_features=4)
        # 5个关键点回归
        self.linear6_3 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(-1, 128 * 3 * 3)
        x = self.linear5(x)
        probs = torch.sigmoid(self.linear6_1(x))
        offset = self.linear6_2(x)
        points = self.linear6_3(x)
        return probs, offset, points


if __name__ == '__main__':
    onet = ONet()
    print(onet)
    X = torch.randn(16, 3, 48, 48)
    print(X.shape)
    probs, offset, points = onet(X)
    print(probs.shape, offset.shape, points.shape)
