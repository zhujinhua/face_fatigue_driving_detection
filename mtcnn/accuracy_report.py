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
from torchvision import transforms

from mtcnn.fast_detect import Detector

SAMPLE_COUNT = 32


class CelebATestDataset(Dataset):
    def __init__(self, img_dir, bbox_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = self._load_data(bbox_file)

    def _load_data(self, bbox_file):
        data = []
        i = 0
        with open(bbox_file, 'r') as f:
            lines = f.readlines()[182639:]
            for line in lines:
                parts = line.strip().split()
                img_name = parts[0]
                bbox = list(map(int, parts[1:]))
                data.append((img_name, bbox))
                i += 1
                if i >= SAMPLE_COUNT:
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


def custom_collate_fn(batch):
    images, bboxes = zip(*batch)
    return list(images), list(bboxes)


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


DATA_ROOT = '/Users/jhzhu/Downloads/software/pan.baidu/CelebA'
test_data_path = os.path.join(DATA_ROOT, 'test')
bbox_file = os.path.join(DATA_ROOT, 'Anno', 'list_bbox_celeba.txt')
img_transfrom = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
test_dataset = CelebATestDataset(img_dir=test_data_path, bbox_file=bbox_file)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = Detector("param_after/p_net.pt", "param_after/r_net.pt", "param_after/o_net.pt")

# 初始化评估指标
all_true_labels = []
all_pred_labels = []

# with torch.no_grad():
#     for images, ground_truth_boxes in test_loader:
#         # add batch detect
#         batch_pnet_boxes, batch_rnet_boxes, batch_onet_boxes = detector.batch_detect(images)
#
#         for i in range(len(batch_onet_boxes)):
#             if batch_onet_boxes[i].ndim == 1:
#                 all_true_labels
#             boxes = batch_onet_boxes[i][:,:4]
#             ground_truth_box = np.array(ground_truth_boxes[i])
#
#             # 计算预测标签和真实标签
#             true_labels = np.zeros(len(ground_truth_box))
#             pred_labels = np.zeros(len(boxes))
#             if len(boxes) == 0:
#                 # 如果没有检测到任何目标，跳过
#                 all_true_labels.extend(true_labels)
#                 all_pred_labels.extend(pred_labels)
#                 continue
#             for j, box in enumerate(boxes):
#                 pred_labels[j] = 0
#                 for k, gt_box in enumerate([ground_truth_box]):
#                     if gt_box.ndim == 1:
#                         continue
#                     if iou(box, gt_box) > 0.5:
#                         pred_labels[j] = 1
#                         true_labels[k] = 1
#
#             all_true_labels.extend(true_labels)
#             all_pred_labels.extend(pred_labels)
with torch.no_grad():
    for images, ground_truth_box in test_loader:
        # 使用 O-Net 进行人脸检测
        batch_pnet_boxes, batch_rnet_boxes, batch_onet_boxes = detector.batch_detect(images)

        for i in range(len(batch_onet_boxes)):
            if batch_onet_boxes[i].ndim == 1:
                pred_label = 0
            else:
                boxes = batch_onet_boxes[i][:, :4]
                gt_box = np.array(ground_truth_box[i])

                # 筛选与真实框IoU最大的预测框
                if len(boxes) == 0:
                    pred_label = 0  # 如果没有预测框
                else:
                    ious = [iou(box, gt_box) for box in boxes]
                    max_iou = max(ious)
                    pred_label = 1 if max_iou > 0.5 else 0  # IoU大于阈值认为匹配正确

            true_label = 1  # 每张图片只有一个真实框，标签为1

            all_true_labels.append(true_label)
            all_pred_labels.append(pred_label)

# 计算评估指标
accuracy = accuracy_score(all_true_labels, all_pred_labels)
precision = precision_score(all_true_labels, all_pred_labels)
recall = recall_score(all_true_labels, all_pred_labels)
f1 = f1_score(all_true_labels, all_pred_labels)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
