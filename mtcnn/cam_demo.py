import cv2

cap = cv2.VideoCapture(0)


while True:
    retval, image = cap.read()
    if retval:
        # 此处插入人脸识别的推理代码


        cv2.imshow(winname="demo", mat=image)
        cv2.waitKey(delay=50)
