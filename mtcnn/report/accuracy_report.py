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

SAMPLE_COUNT = 100
IOU_LIST = [0.5, 0.7, 0.8, 0.9, 0.95, 1]


# IOU_LIST = [0.95]


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


def draw_rectangle(img, output_path, boxes, color=(0, 0, 255)):
    image = cv2.imread(img.filename)
    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=3)
    cv2.imwrite(output_path, image)


def plot_accuracies(iou_thresholds, accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(iou_thresholds, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
    plt.xlabel('IoU Thresholds')
    plt.ylabel('Accuracy')
    plt.title('Face Detection Accuracy at Different IoU Thresholds')
    plt.ylim([0, 1])
    plt.grid(True)

    # Annotate each point with the accuracy value
    for i, acc in enumerate(accuracies):
        plt.text(iou_thresholds[i], acc + 0.02, f'{acc:.3f}', ha='center', va='bottom')

    plt.legend()
    plt.show()


detect_result = './test_result'
more_result = './more_result'

DATA_ROOT = '/Users/jhzhu/Downloads/software/pan.baidu/CelebA'
test_data_path = os.path.join(DATA_ROOT, 'test')
bbox_file = os.path.join(DATA_ROOT, 'Anno', 'list_bbox_celeba.txt')
test_dataset = CelebATestDataset(img_dir=test_data_path, bbox_file=bbox_file)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = Detector("../param/p_net.pt", "../param/r_net.pt", "../param/o_net.pt")

# Processing test data once
all_true_labels = []
all_pred_labels_dict = {iou_thresh: [] for iou_thresh in IOU_LIST}

with torch.no_grad():
    for images, ground_truth_box in test_loader:
        batch_pnet_boxes, batch_rnet_boxes, batch_onet_boxes = detector.batch_detect(images)

        for i in range(len(batch_onet_boxes)):
            if batch_onet_boxes[i].ndim == 1:
                for iou_thresh in IOU_LIST:
                    all_pred_labels_dict[iou_thresh].append(0)
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

                if len(boxes) != 1:
                    for iou_thresh in IOU_LIST:
                        all_pred_labels_dict[iou_thresh].append(0)
                else:
                    ious = [iou(box, [x1, y1, x2, y2]) for box in boxes]
                    max_iou = max(ious)
                    for iou_thresh in IOU_LIST:
                        pred_label = 1 if max_iou > iou_thresh else 0  # IoU大于阈值认为匹配正确
                        all_pred_labels_dict[iou_thresh].append(pred_label)
                        # draw_rectangle(images[i], os.path.join(detect_result, '%s_%s' % (iou_thresh, os.path.basename(images[i].filename))), boxes)

            true_label = 1  # 每张图片只有一个真实框，标签为1
            all_true_labels.append(true_label)

# Calculate accuracy for each IoU threshold
accuracies = [accuracy_score(all_true_labels, all_pred_labels_dict[iou_thresh]) for iou_thresh in IOU_LIST]

# Plot the accuracies
plot_accuracies(IOU_LIST, accuracies)
