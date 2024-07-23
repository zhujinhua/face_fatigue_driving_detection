# 1，引入 YOLO 类
from ultralytics import YOLO
import cv2

# 2，加载预训练权重
# 1.0版本生成的权重
# model = YOLO('best.pt')
# 2.0版本生成的权重
model = YOLO('best2.pt')

# 建立连接
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 3，读取视频帧
    # 读取一帧，返回：读取状态，图像数据
    status, frame = cap.read()
    
    if not status:
        print('读取失败')
        break

    # 4，模型预测
    results = model(frame)

    # 5，绘制预测结果
    img = results[0].plot()

    # 6，显示图像
    cv2.imshow('frame', img)
    
    # 按键等待
    if cv2.waitKey(int(1000/24)) == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()