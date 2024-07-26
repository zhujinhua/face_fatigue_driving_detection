"""
Author: jhzhu
Date: 2024/7/26
Description: 
"""
import os

from mtcnn.api.mtcnn_detect import get_detect_face

img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api', '10.jpg')
cropped_img_tensor = get_detect_face(img_path, target_size=(128, 128))
# [channels, height, weight]
print(cropped_img_tensor.shape)