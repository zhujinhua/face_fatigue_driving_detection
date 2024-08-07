"""
Author: jhzhu
Date: 2024/8/5
Description: 
"""
import cv2
import numpy as np


# 读取图像
image = cv2.imread('./detect_img/kang_4.jpg')

# 高斯滤波去噪
denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
# 保存增强后的图像
cv2.imwrite('detect_img/enhanced_face_image.jpg', denoised_image)

