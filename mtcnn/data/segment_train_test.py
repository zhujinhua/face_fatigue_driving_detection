"""
Author: jhzhu
Date: 2024/7/25
Description: 
"""
import os
import shutil

# 数据集路径
dataset_dir = '/Users/jhzhu/Downloads/software/pan.baidu/CelebA'
image_dir = os.path.join(dataset_dir, 'Img', 'img_celeba')
partition_file = os.path.join(dataset_dir, 'Eval', 'list_eval_partition.txt')

# 目标路径
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

# 创建目标文件夹
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 解析分区文件
with open(partition_file, 'r') as file:
    lines = file.readlines()
    for line in lines:
        image_name, partition = line.strip().split()
        image_path = os.path.join(image_dir, image_name)

        if partition == '0':  # 0表示训练集
            shutil.copy(image_path, train_dir)
        elif partition == '2':  # 2表示测试集
            shutil.copy(image_path, test_dir)

print("数据集划分完成！")

