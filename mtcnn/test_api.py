"""
Author: jhzhu
Date: 2024/7/26
Description: 
"""
import os
import unittest
import cv2
import torch
from PIL import Image

from mtcnn.api.mtcnn_detect import get_detect_face, get_mtcnn_detector
from mtcnn.api.tool import resize_image


class TestAPI(unittest.TestCase):
    def test_get_detect_face(self):
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api', '10.jpg')
        cropped_img_tensor = get_detect_face(img_path, target_size=(128, 128))
        # [channels, height, weight]
        assert cropped_img_tensor.shape == torch.Size([3, 128, 128])

    def test_detect_on_camera(self):
        output_video_path = './output_video.mp4'
        detector = get_mtcnn_detector()
        cap = cv2.VideoCapture(0)
        # Get size and fps of video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Create VideoWriter for saving
        outVideo = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        c = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if c % 1 == 0:
                    # BGR covert to RGB, for mtcnn used RGB
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    # img = resize_image(img, 512)
                    # img.save('frame.jpg')
                    detect_boxes = detector.detect(img)
                    if len(detect_boxes) == 0:
                        onet_boxes = detect_boxes
                    else:
                        pnet_boxes, rnet_boxes, onet_boxes = detect_boxes
                    if onet_boxes.shape[0] != 0:
                        # image = cv2.imread('frame.jpg')
                        for box in onet_boxes:
                            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 255, 0), thickness=25)
                            text = f'{box[4]:.2f}'
                            cv2.putText(frame, text, (int(box[0]), int(box[1]) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 5)
                            # cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 255, 0), thickness=25)
                        # cv2.imwrite('./check.jpg', image)
                cv2.imshow('video', frame)
                outVideo.write(frame)
                if cv2.waitKey(1) == 27:
                    break
            c += 1
        cap.release()
        cv2.destroyAllWindows()