# 引入PyTorch
import torch
from torch import nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // 16)
        self.fc2 = nn.Linear(in_channels // 16, in_channels)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        max_pool = F.adaptive_max_pool2d(x, 1).view(batch_size, channels)
        avg_out = F.relu(self.fc1(avg_pool))
        avg_out = self.fc2(avg_out).view(batch_size, channels, 1, 1)
        max_out = F.relu(self.fc1(max_pool))
        max_out = self.fc2(max_out).view(batch_size, channels, 1, 1)
        out = torch.sigmoid(avg_out + max_out)
        return x * out


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
        # add attention
        self.channel_attention = ChannelAttention(32)
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
        x = self.channel_attention(x)
        # [N, 32, 1, 1] --> [N, 1, 1, 1]
        probs = torch.sigmoid(self.conv4_1(x))
        # [N, 32, 1, 1] --> [N, 4, 1, 1]
        offset = self.conv4_2(x)
        return probs, offset


class FeatureFusion(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(FeatureFusion, self).__init__()
        self.conv1x1_x1 = nn.Conv2d(in_channels1, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1x1_x2 = nn.Conv2d(in_channels2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        # 调整通道数
        x1 = self.conv1x1_x1(x1)
        x2 = self.conv1x1_x2(x2)

        # 使特征图具有相同的空间尺寸
        if x1.size(2) != x2.size(2) or x1.size(3) != x2.size(3):
            x2 = F.interpolate(x2, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=False)

        # 融合特征图
        x_fused = torch.cat((x1, x2), dim=1)  # 拼接通道维度
        return x_fused


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.feature_extractor_stage1 = nn.Sequential(
            nn.Conv2d(3, 28, 3, 1, 0),
            nn.BatchNorm2d(28),
            nn.PReLU(28),
            nn.MaxPool2d(3, 2, 1),
        )
        self.feature_extractor_stage2 = nn.Sequential(
            nn.Conv2d(28, 48, 3, 1, 0),
            nn.BatchNorm2d(48),
            nn.PReLU(48),
            nn.MaxPool2d(3, 2, 0),
        )
        self.feature_extractor_stage3 = nn.Sequential(
            nn.Conv2d(48, 64, 2, 1, 0),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        # 初始化FeatureFusion类，处理不同尺度特征图
        self.feature_fusion_stage1_2 = FeatureFusion(28, 48, 64)
        self.feature_fusion_stage2_3 = FeatureFusion(128, 64, 128)
        self.channel_attention = ChannelAttention(256)
        self.linear4 = nn.Sequential(
            nn.Linear(in_features=256*11*11, out_features=128),
            nn.PReLU(128)
        )

        # 类别
        self.linear5_1 = nn.Linear(in_features=128, out_features=1)
        # 误差回归
        self.linear5_2 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x):
        x1 = self.feature_extractor_stage1(x)
        x2 = self.feature_extractor_stage2(x1)
        x3 = self.feature_extractor_stage3(x2)

        # Feature fusion with different scales
        x_fused = self.feature_fusion_stage1_2(x1, x2)
        x_fused = self.feature_fusion_stage2_3(x_fused, x3)

        x_fused = self.channel_attention(x_fused)
        x_fused = self.linear4(x_fused.view(x_fused.size(0), -1))
        probs = torch.sigmoid(self.linear5_1(x_fused))
        offset = self.linear5_2(x_fused)
        return probs, offset


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.feature_extractor_stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            nn.MaxPool2d(3, 2, 1),
        )
        self.feature_extractor_stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 0),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.MaxPool2d(3, 2, 0),
        )
        self.feature_extractor_stage3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.MaxPool2d(2, 2, 0),
        )
        self.feature_extractor_stage4 = nn.Sequential(
            nn.Conv2d(64, 128, 2, 1, 0),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
        )
        self.feature_fusion_stage1_2 = FeatureFusion(32, 64, 64)
        self.feature_fusion_stage3_4 = FeatureFusion(128, 64, 128)
        self.channel_attention = ChannelAttention(128)
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
        x1 = self.pre_layer(x)
        x2 = x1
        x = self.feature_fusion(x1, x2)
        x = self.channel_attention(x)
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
