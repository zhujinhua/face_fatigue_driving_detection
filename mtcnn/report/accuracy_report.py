"""
Author: jhzhu
Date: 2024/7/28
Description: 
"""
from mtcnn.api.mtcnn_detect import Detector

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

SAMPLE_COUNT = 4000
# IOU_LIST = [0.5, 0.7, 0.8, 0.9, 0.95]
IOU_LIST = [0.95]


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


detect_result = './'
DATA_ROOT = '/Users/jhzhu/Downloads/software/pan.baidu/CelebA'
test_data_path = os.path.join(DATA_ROOT, 'test')
bbox_file = os.path.join(DATA_ROOT, 'Anno', 'list_bbox_celeba.txt')
test_dataset = CelebATestDataset(img_dir=test_data_path, bbox_file=bbox_file)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = Detector("../param/p_net.pt", "../param/r_net.pt", "../param/o_net.pt")

for _iou in IOU_LIST:
    all_true_labels = []
    all_pred_labels = []

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
                    x = int(gt_box[0])
                    y = int(gt_box[1])
                    w = int(gt_box[2])
                    h = int(gt_box[3])
                    x1 = int(x + w * 0.12)
                    y1 = int(y + h * 0.1)
                    x2 = int(x + w * 0.9)
                    y2 = int(y + h * 0.85)
                    # 筛选与真实框IoU最大的预测框
                    if len(boxes) == 0:
                        pred_label = 0  # 如果没有预测框
                    else:
                        ious = [iou(box, [x1, y1, x2, y2]) for box in boxes]
                        max_iou = max(ious)
                        pred_label = 1 if max_iou > _iou else 0  # IoU大于阈值认为匹配正确

                true_label = 1  # 每张图片只有一个真实框，标签为1

                all_true_labels.append(true_label)
                all_pred_labels.append(pred_label)

    # 计算评估指标
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    # precision = precision_score(all_true_labels, all_pred_labels)
    # recall = recall_score(all_true_labels, all_pred_labels)
    # f1 = f1_score(all_true_labels, all_pred_labels)

    print(f'IOU: {_iou}, Accuracy: {accuracy:.4f}')
    # print(f'IOU: {i}, Precision: {precision:.4f}')
    # print(f'IOU: {i}, Recall: {recall:.4f}')
    # print(f'IOU: {i}, F1 Score: {f1:.4f}')

