"""
Author: jhzhu
Date: 2024/7/25
Description: Use the test data to get the accuracy, test data samples 4000
"""
import os

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from PIL import Image
from mtcnn.detect import Detector
from mtcnn.tool import iou

SAMPLE_COUNT = 4000


class CelebATestDataset(Dataset):
    def __init__(self, img_dir, bbox_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = self._load_data(bbox_file)

    def _load_data(self, bbox_file):
        data = []
        i = 0
        with open(bbox_file, 'r') as f:
            lines = f.readlines()[2:]  # 跳过前两行
            for line in lines:
                parts = line.strip().split()
                img_name = parts[0]
                bbox = list(map(int, parts[1:]))
                data.append((img_name, bbox))
                i += 1
                if i >= 4000:
                    break
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, bbox = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, bbox


DATA_ROOT = '/Users/jhzhu/Downloads/software/pan.baidu/CelebA'
test_data_path = os.path.join(DATA_ROOT, 'test')
bbox_file = os.path.join(DATA_ROOT, 'Anno', 'list_bbox_celeba.txt')
test_dataset = CelebATestDataset(img_dir=test_data_path, bbox_file=bbox_file)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = Detector("param/p_net.pt", "param/r_net.pt", "param/o_net.pt")

# 初始化评估指标
all_true_labels = []
all_pred_labels = []

with torch.no_grad():
    for images, ground_truth_boxes in test_loader:
        images = images.to(device)
        # add batch detect
        batch_pnet_boxes, batch_rnet_boxes, batch_onet_boxes = detector.detect(images)

        for i in range(len(batch_onet_boxes)):
            boxes = batch_onet_boxes[i]
            ground_truth_box = ground_truth_boxes[i].numpy()

            # 计算预测标签和真实标签
            true_labels = np.zeros(len(ground_truth_box))
            pred_labels = np.zeros(len(boxes))

            for j, box in enumerate(boxes):
                for k, gt_box in enumerate([ground_truth_box]):
                    if iou(box, gt_box) > 0.5:
                        pred_labels[j] = 1
                        true_labels[k] = 1

            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)

# 计算评估指标
accuracy = accuracy_score(all_true_labels, all_pred_labels)
precision = precision_score(all_true_labels, all_pred_labels)
recall = recall_score(all_true_labels, all_pred_labels)
f1 = f1_score(all_true_labels, all_pred_labels)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
