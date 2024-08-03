"""
Author: jhzhu
Date: 2024/8/2
Description: 
"""
import os.path

import albumentations as A
import numpy as np
import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform


class AddWhiteNoise(ImageOnlyTransform):
    def __init__(self, noise_factor=0.02, always_apply=False, p=0.5):
        super(AddWhiteNoise, self).__init__(always_apply=always_apply, p=p)
        self.noise_factor = noise_factor

    def apply(self, img, **params):
        noise = np.random.rand(*img.shape[:2]) < self.noise_factor
        img[noise] = 255
        return img


if __name__ == '__main__':
    # 创建一个增强管道，包括添加白噪声的变换
    transform = A.Compose([
        AddWhiteNoise(noise_factor=0.25, always_apply=False, p=1),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0), contrast_limit=0, p=1),
        # A.HorizontalFlip(p=0.5),
        # A.Rotate(limit=30, p=0.5),
    ])

    # 读取图像
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detect_img', '01.jpg')
    image = cv2.imread(image_path)

    # 应用增强变换
    augmented = transform(image=image)
    augmented_image = augmented['image']

    # 显示和保存增强后的图像
    cv2.imshow('Augmented Image', augmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('../augmented_image.jpg', augmented_image)
