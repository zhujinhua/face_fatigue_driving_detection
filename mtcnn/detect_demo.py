from PIL import Image
import cv2
import os
from fast_detect import Detector
from matplotlib import pyplot as plt

param_path = "param1"
p_net = os.path.join(param_path, "p_net.pt")
r_net = os.path.join(param_path, "r_net.pt")
o_net = os.path.join(param_path, "o_net.pt")

img_name = r"02.jpg"
detect_img = "./data/detect_img"
out_img = "./data/out_img"

if not os.path.exists("./data/out_img"):
    os.makedirs("./data/out_img", exist_ok=True)

if __name__ == '__main__':

    # 合成全路径
    img_path = os.path.join(detect_img, img_name)

    # 读取图像
    img = Image.open(img_path)

    # 加载模型
    detector = Detector(p_net, r_net, o_net,
                        softnms=False,
                        thresholds=[0.6, 0.7, 0.95])

    # 人脸检测
    detect_boxes = detector.detect(img)

    # 绘制结果
    if len(detect_boxes) == 0:
        onet_boxes = detect_boxes
    else:
        pnet_boxes, rnet_boxes, onet_boxes = detect_boxes
        print("pnet:", pnet_boxes.shape)
        print("rnet:", rnet_boxes.shape)
        print("onet:", onet_boxes.shape)

    img = cv2.imread(img_path)
    if onet_boxes.shape[0] != 0:
        for box in onet_boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(img, (x1, y1), (x2, y2),
                          color=(0, 0, 255), thickness=2)
            for i in range(5, 15, 2):
                cv2.circle(img, (int(box[i]), int(box[i + 1])),
                           radius=1, color=(255, 255, 0), thickness=-1)
    # cv2.imwrite(os.path.join(out_img, img_name), img)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    plt.imshow(X=img[:, :, ::-1])
    plt.show()
