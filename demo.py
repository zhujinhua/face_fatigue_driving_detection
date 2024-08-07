#! pip install pillow
import os.path
import time
import platform

from faceNet.face_api import face_validation
from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# 加载中文字体
def get_the_system_font():
    if platform.system() == 'Windows':
        font_path = 'C:/Windows/Fonts/simhei.ttf'  # Windows 的中文字体路径
        font = ImageFont.truetype(font_path, 30)
    elif platform.system() == 'Darwin':
        font_path = '/System/Library/Fonts/STHeiti Medium.ttc'  # macOS 的中文字体路径
        font = ImageFont.truetype(font_path, 50)
    else:
        raise OSError("Unsupported operating system")
    return font


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
font = get_the_system_font()
model = YOLO(os.path.join(CUR_DIR, 'fatigue_driving', 'best.pt'))

cap = cv2.VideoCapture(0)
num = 0
start_time = None
elapsed_time = 0
closed_eyes_duration = 10
closed_eyes_detected = False
tips = '您已疲劳驾驶, 车辆将开启自动驾驶模式!!!'
while cap.isOpened():
    status, frame = cap.read()
    num += 1
    if not status:
        print('读取摄像头失败')
        break

    if num % 1 == 0:
        # BGR转RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.GaussianBlur(frame_rgb, (5, 5), 0)
        # 1. 使用Pillow读取图像
        image = Image.fromarray(frame_rgb)
        # image.save('output_image3.png')

        # 2. 人脸检测并识别,res返回非None,识别成功
        res, box = face_validation(img=image)
        va_text = ' Verification Successful'
        if not res:
            va_text = ' Verification Failed'
        print(va_text)
        # 3. 画出检测到的人脸框
        if len(box) > 0:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 255, 0),
                          thickness=8)
            text = f'{box[4]:.2f}'
            cv2.putText(frame, text + va_text, (int(box[0]), int(box[1]) - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 8)
        # 4. 疲劳驾驶检测
        results = model(frame)
        for result in results:
            for obj in result.boxes:
                # 假设闭眼类别的 ID 是 1，根据你的实际情况修改
                if obj.cls == 3 or obj.cls == 0:
                    closed_eyes_detected = True
                    break

        # 更新计时器
        if closed_eyes_detected:
            if start_time is None:
                start_time = time.time()  # 开始计时
            elapsed_time = time.time() - start_time
        else:
            start_time = None  # 重置计时器
            elapsed_time = 0

        # 绘制预测结果并添加提示文字
        img = results[0].plot()
        if elapsed_time >= closed_eyes_duration:
            closed_eyes_detected = False
            # 中文字体，需要用使用PIL
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            # 计算文本大小
            bbox = draw.textbbox((0, 0), tips, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # 定义文本位置（在底部中央）
            image_width, image_height = img_pil.size
            x = (image_width - text_width) / 2
            y = image_height - text_height - 20  # 10像素的底部边距
            # 绘制文本
            draw.text((x, y), tips, font=font, fill=(255, 0, 0))
            # 将PIL图像转换回OpenCV图像
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            print(tips)
            # cv2.imwrite('image_with_text.jpg', img)

        # 5. 显示图像
        cv2.imshow('frame', img)

    # 6. 按键等待
    if cv2.waitKey(int(1000 / 24)) == 27:
        break

# 7. 释放资源
cap.release()
cv2.destroyAllWindows()
