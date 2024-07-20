from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import os
from PIL import Image


class FaceDataset(Dataset):
    """
        打包数据成标准数据集
    """

    def __init__(self, path):
        super(FaceDataset, self).__init__()
        self.path = path
        self.datasets = []
        self._read_annos()

    def _read_annos(self):
        with open(os.path.join(self.path, "positive.txt")) as f:
            self.datasets.extend(f.readlines())

        with open(os.path.join(self.path, "negative.txt")) as f:
            self.datasets.extend(f.readlines())

        with open(os.path.join(self.path, "part.txt")) as f:
            self.datasets.extend(f.readlines())

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        strs = self.datasets[idx].strip().split()
        # 文件名字
        img_name = strs[0]

        # 取出类别
        cls = torch.tensor([int(strs[1])], dtype=torch.float32)

        # 将所有偏置转为float类型
        strs[2:] = [float(x) for x in strs[2:]]

        # bbox的偏置
        offset = torch.tensor(strs[2:6], dtype=torch.float32)
        # landmark的偏置
        point = torch.tensor(strs[6:16], dtype=torch.float32)

        # 打开图像
        img = Image.open(os.path.join(self.path, img_name))

        # 数据调整到 [-1, 1]之间
        img_data = torch.tensor((np.array(img) / 255. - 0.5) / 0.5, dtype=torch.float32)
        # [H, W, C] --> [C, H ,W]
        img_data = img_data.permute(2, 0, 1)

        return img_data, cls, offset, point


if __name__ == '__main__':
    data = FaceDataset(r"D:\DataSet\MTCNN\48")
    print(data[0][0].shape)
    print(data[0][1])
    print(data[0][2])
    print(data[0][3])

    print(len(data))
    dataloder = DataLoader(data, batch_size=5, shuffle=True)
    print(len(dataloder))
    for img_data, cls, offset, point in dataloder:
        print(img_data.shape)
        print(cls)
        print(offset)
        print(point)
        break
