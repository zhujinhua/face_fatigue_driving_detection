#! pip install pillow
from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 2，加载预训练权重
model = YOLO('best.pt')

# 建立连接
cap = cv2.VideoCapture(0)

# 加载中文字体
font_path = 'C:/Windows/Fonts/simhei.ttf'    # 替换为你的中文字体路径
font = ImageFont.truetype(font_path, 24)

while cap.isOpened():
    # 3，读取视频帧
    status, frame = cap.read()
    
    if not status:
        print('读取失败')
        break

    # 4，模型预测
    results = model(frame)
    
    # 5，检查是否检测到闭眼
    closed_eyes_detected = False
    for result in results:
        for obj in result.boxes:
            # 假设闭眼类别的 ID 是 1，根据你的实际情况修改
            if obj.cls == 3 or obj.cls == 0:
                closed_eyes_detected = True
                break
    
    # 6，绘制预测结果并添加提示文字
    img = results[0].plot()
    if closed_eyes_detected:
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((50, 50), '您已疲劳驾驶！车辆已开启自动驾驶。', font=font, fill=(255, 0, 0))

        # 将PIL图像转换回OpenCV图像
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 7，显示图像
    cv2.imshow('frame', img)
    
    # 按键等待
    if cv2.waitKey(int(1000/24)) == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
