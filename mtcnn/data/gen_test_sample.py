"""
Author: jhzhu
Date: 2024/8/3
Description: 
"""
import os

from PIL import Image

os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

DATA_ROOT = '/Users/jhzhu/Downloads/software/pan.baidu/CelebA'
# 图像
img_dir = os.path.join(DATA_ROOT, 'test')
# 框的标注
anno_src = os.path.join(DATA_ROOT, 'Anno/list_bbox_celeba.txt')
# 关键点的标注
anno_landmarks_src = os.path.join(DATA_ROOT, 'Anno/list_landmarks_celeba.txt')
# 结果保存
save_dir = r"../test_data/MTCNN"


def gen_test_sample(face_size, stop_value):
    """
        face_size: 图像大小，12， 24， 48
        stop_value：图像总的个数
    """
    positive_img_dir = os.path.join(save_dir, str(face_size), "positive")
    if not os.path.exists(positive_img_dir):
        os.makedirs(positive_img_dir)
    # 创建保存标签的文件，并打开文件
    anno_positive_filename = os.path.join(save_dir, str(face_size), "positive.txt")

    try:
        # 新建标注文件
        anno_positive_file = open(anno_positive_filename, 'w')
        # 样本计数
        positive_count = 0

        # 按行读取5个关键点的标签文件，返回一个列表【关键点】
        with open(anno_landmarks_src) as f:
            landmarks_list = f.readlines()

        # 读取CelebA的标签文件【框的信息】
        with open(anno_src) as f:
            anno_list = f.readlines()

        # 打开人脸框的标签，循环读取每一行
        for i, (anno_line, landmarks) in enumerate(zip(anno_list, landmarks_list)):

            # 跳过表头
            if i < 182639:
                continue

            # 5个关键点
            landmarks = landmarks.split()
            # 定位框
            strs = anno_line.split()

            # 解析文件名字
            img_name = strs[0].strip()

            # 读取图像
            img = Image.open(os.path.join(img_dir, img_name))

            # 转换框坐标的类型
            x, y, w, h = float(strs[1].strip()), float(strs[2].strip()), float(strs[3].strip()), float(strs[4].strip())

            # 标签矫正[不管]
            x1 = int(x + w * 0.12)
            y1 = int(y + h * 0.1)
            x2 = int(x + w * 0.9)
            y2 = int(y + h * 0.85)

            # 计算新的宽度和高度
            w, h = x2 - x1, y2 - y1
            # 判断坐标是否符合要求
            if max(w, h) < 40 or x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                continue

            # 接下来，开始生成样本
            box = [x1, y1, x2, y2]
            img_crop = img.crop(box)
            img_crop = img_crop.resize((face_size, face_size))
            img_crop.save(os.path.join(positive_img_dir, "{0}.jpg".format(positive_count)))
            anno_positive_file.write(
                "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                    positive_count, 1,
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0))
            anno_positive_file.flush()
            positive_count += 1
            if positive_count > stop_value:
                break
    except Exception as ex:
        print(ex)


if __name__ == '__main__':
    # 4000
    # P-Net
    gen_test_sample(12, 10000)
    # R-Net
    gen_test_sample(24, 10000)
    # O-Net
    gen_test_sample(48, 10000)
