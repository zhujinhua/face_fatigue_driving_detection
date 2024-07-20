import os
import numpy as np
import tool
from PIL import Image
import traceback
import os

DATA_ROOT = '/Users/jhzhu/Downloads/software/pan.baidu/CelebA'
# 图像
img_dir = os.path.join(DATA_ROOT, 'Img/img_celeba')
# 框的标注
anno_src = os.path.join(DATA_ROOT, 'Anno/list_bbox_celeba.txt')
# 关键点的标注
anno_landmarks_src = os.path.join(DATA_ROOT, 'Anno/list_landmarks_celeba.txt')
# 结果保存
save_dir = r"../test_data/MTCNN"

# 为随机数种子做准备，使正样本，部分样本，负样本的比例为1：1：3
float_num = [0.1, 0.1, 0.3, 0.5, 0.95, 0.95, 0.99, 0.99, 0.99, 0.99]


def gen_sample(face_size, stop_value):
    """
        face_size: 图像大小，12， 24， 48
        stop_value：图像总的个数
    """

    # 创建保存样本的目录（正，负，偏）
    positive_img_dir = os.path.join(save_dir, str(face_size), "positive")
    negative_img_dir = os.path.join(save_dir, str(face_size), "negative")
    part_img_dir = os.path.join(save_dir, str(face_size), "part")
    for dir_path in [positive_img_dir, negative_img_dir, part_img_dir]:
        if not os.path.exists(dir_path):
            # 递归创建目录
            os.makedirs(dir_path)

    # 创建保存标签的文件，并打开文件
    anno_positive_filename = os.path.join(save_dir, str(face_size), "positive.txt")
    anno_negative_filename = os.path.join(save_dir, str(face_size), "negative.txt")
    anno_part_filename = os.path.join(save_dir, str(face_size), "part.txt")

    try:
        # 新建标注文件
        anno_positive_file = open(anno_positive_filename, 'w')
        anno_negative_file = open(anno_negative_filename, 'w')
        anno_part_file = open(anno_part_filename, 'w')

        # 样本计数
        positive_count = 0
        negative_count = 0
        part_count = 0

        # 按行读取5个关键点的标签文件，返回一个列表【关键点】
        with open(anno_landmarks_src) as f:
            landmarks_list = f.readlines()

        # 读取CelebA的标签文件【框的信息】
        with open(anno_src) as f:
            anno_list = f.readlines()

        # 打开人脸框的标签，循环读取每一行
        for i, (anno_line, landmarks) in enumerate(zip(anno_list, landmarks_list)):

            # 跳过表头
            if i < 2:
                continue

            # 5个关键点
            landmarks = landmarks.split()
            # 定位框
            strs = anno_line.split()

            # 解析文件名字
            img_name = strs[0].strip()

            # 读取图像
            img = Image.open(os.path.join(img_dir, img_name))

            # 解析出宽度和高度
            img_w, img_h = img.size

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

            # 记录5个关键点的坐标
            px1 = float(landmarks[1].strip())
            py1 = float(landmarks[2].strip())
            px2 = float(landmarks[3].strip())
            py2 = float(landmarks[4].strip())
            px3 = float(landmarks[5].strip())

            py3 = float(landmarks[6].strip())
            px4 = float(landmarks[7].strip())
            py4 = float(landmarks[8].strip())
            px5 = float(landmarks[9].strip())
            py5 = float(landmarks[10].strip())

            # 接下来，开始生成样本
            box = [x1, y1, x2, y2]

            # 求出中心点和边长，偏移中心点和边长得到样本，每张图偏移5次
            cx = x1 + w / 2
            cy = y1 + h / 2

            # 最大边长
            max_side = max(w, h)

            # 尝试5次
            for _ in range(5):

                # 为随机数种子做准备，使正样本，部分样本，负样本的比例为1：1：3
                # float_num = [0.1, 0.1, 0.3, 0.5, 0.95, 0.95, 0.99, 0.99, 0.99, 0.99]

                # 随机偏移中心点坐标以及边长
                seed = float_num[np.random.randint(0, len(float_num))]

                # 最大边长随机偏移
                _max_side = max_side + np.random.randint(int(-max_side * seed), int(max_side * seed))

                # 中心点x坐标随机偏移
                _cx = cx + np.random.randint(int(-cx * seed), int(cx * seed))

                # 中心点y坐标随机偏移
                _cy = cy + np.random.randint(int(-cy * seed), int(cy * seed))

                # 得到偏移后的坐标值（方框）
                _x1 = _cx - _max_side / 2
                _y1 = _cy - _max_side / 2
                _x2 = _x1 + _max_side
                _y2 = _y1 + _max_side

                # 偏移过大，偏出图像了，此时，不能用，应该再次尝试偏移
                if _x1 < 0 or _y1 < 0 or _x2 > img_w or _y2 > img_h:
                    continue

                # 记录偏移后的坐标
                cbox = [_x1, _y1, _x2, _y2]

                # --------------
                # 计算两个坐标点和5个关键点的偏移率
                # （老 - 新） / 新的最大边长
                # 误差是人为构建出来的，可正向使用，也可以反向使用推导出原来的坐标，可以任意设置
                offset_x1 = (x1 - _x1) / _max_side
                offset_y1 = (y1 - _y1) / _max_side
                offset_x2 = (x2 - _x2) / _max_side
                offset_y2 = (y2 - _y2) / _max_side

                offset_px1 = (px1 - _x1) / _max_side
                offset_py1 = (py1 - _y1) / _max_side
                offset_px2 = (px2 - _x1) / _max_side
                offset_py2 = (py2 - _y1) / _max_side
                offset_px3 = (px3 - _x1) / _max_side
                offset_py3 = (py3 - _y1) / _max_side
                offset_px4 = (px4 - _x1) / _max_side
                offset_py4 = (py4 - _y1) / _max_side
                offset_px5 = (px5 - _x1) / _max_side
                offset_py5 = (py5 - _y1) / _max_side

                # 根据偏移后的坐标截图图片，并缩放成要训练的大小
                img_crop = img.crop(cbox)
                img_crop = img_crop.resize((face_size, face_size))

                # 对偏移框和真实框做iou, 根据偏离程度划分样本
                iou = tool.iou(box, np.array([cbox]))[0]

                if iou > 0.7:
                    img_crop.save(os.path.join(positive_img_dir, "{0}.jpg".format(positive_count)))
                    anno_positive_file.write(
                        "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                            positive_count, 1,
                            offset_x1, offset_y1, offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2,
                            offset_px3, offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                    anno_positive_file.flush()
                    positive_count += 1
                elif 0.4 < iou < 0.6:
                    img_crop.save(os.path.join(part_img_dir, "{0}.jpg".format(part_count)))
                    anno_part_file.write(
                        "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                            part_count, 2,
                            offset_x1, offset_y1, offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2,
                            offset_px3, offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                    anno_part_file.flush()
                    part_count += 1
                elif iou < 0.2:
                    img_crop.save(os.path.join(negative_img_dir, "{0}.jpg".format(negative_count)))
                    anno_negative_file.write("negative/{0}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count))
                    anno_negative_file.flush()
                    negative_count += 1
            count = positive_count + negative_count + part_count
            if count > stop_value:
                break
    except Exception as ex:
        print(ex)


if __name__ == '__main__':
    # 100000
    # P-Net
    gen_sample(12, 5000)
    # R-Net
    gen_sample(24, 5000)
    # O-Net
    gen_sample(48, 5000)
