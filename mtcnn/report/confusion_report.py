"""
Author: jhzhu
Date: 2024/7/25
Description: Use the test data to get the accuracy, test data samples 4000
"""
import os

import cv2

from mtcnn.api.mtcnn_detect import Detector
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


SAMPLE_COUNT = 4000
output_dir = './test_result'


class CelebATestDataset(Dataset):
    def __init__(self, img_dir, bbox_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = self._load_data(bbox_file)

    def _load_data(self, bbox_file):
        data = []
        i = 0
        with open(bbox_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                img_name = parts[0]
                label = parts[1]
                bbox = list(map(int, parts[2:]))
                data.append((img_name, label, bbox))
                i += 1
                if i >= SAMPLE_COUNT:
                    break
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label, bbox = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label, bbox


def custom_collate_fn(batch):
    images, labels, bboxes = zip(*batch)
    return list(images), list(labels), list(bboxes)


def iou(box, gt_box):
    # 计算两个框的交并比 (IoU)
    x1, y1, x2, y2 = box
    x1g, y1g, x2g, y2g = gt_box

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)

    box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    gt_box_area = (x2g - x1g + 1) * (y2g - y2g + 1)
    union_area = box_area + gt_box_area - inter_area

    iou = inter_area / union_area
    return iou


detect_result = './'
DATA_ROOT = '/Users/jhzhu/code_repository/git_project/face_fatigue_driving_detection/mtcnn/negative_data'
test_data_path = os.path.join(DATA_ROOT, 'image')
bbox_file = os.path.join(DATA_ROOT, 'bbox.txt')
test_dataset = CelebATestDataset(img_dir=test_data_path, bbox_file=bbox_file)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = Detector("../param/p_net.pt", "../param/r_net.pt", "../param/o_net.pt")

all_true_labels = []
all_pred_labels = []

with torch.no_grad():
    for images, labels, ground_truth_box in test_loader:
        # 使用 O-Net 进行人脸检测
        detect_boxes = detector.batch_detect(images)
        if len(detect_boxes) == 0:
            batch_onet_boxes = detect_boxes
        else:
            batch_pnet_boxes, batch_rnet_boxes, batch_onet_boxes = detect_boxes

        for i in range(len(batch_onet_boxes)):
            all_true_labels.append(int(labels[i]))
            if batch_onet_boxes[i].ndim == 1:
                # 没有检测出人脸，不添加真实标签和预测标签，继续下一个样本
                all_pred_labels.append(0)
                continue
            else:
                boxes = batch_onet_boxes[i][:, :4]
                gt_box = np.array(ground_truth_box[i])
                x = int(gt_box[0])
                y = int(gt_box[1])
                w = int(gt_box[2])
                h = int(gt_box[3])
                x1 = int(x + w * 0.12)
                y1 = int(y + h * 0.1)
                x2 = int(x + w * 0.9)
                y2 = int(y + h * 0.85)
                # None Face 0 ->Face 1
                if len(boxes) == 0:
                    pred_label = 0
                else:
                    ious = [iou(box, [x1, y1, x2, y2]) for box in boxes]
                    max_iou = max(ious)
                    pred_label = 1 if max_iou > 0.95 else 0
                if pred_label == 0 and labels[i] == 1:
                    cv2.imwrite(os.path.join(output_dir, os.path.basename(images[i].filename)), images[i])

                all_pred_labels.append(pred_label)

# 计算混淆矩阵
cm = confusion_matrix(all_true_labels, all_pred_labels)

# 绘制混淆矩阵热图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Face', 'Face'],
            yticklabels=['Non-Face', 'Face'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Face Detection')
plt.show()

