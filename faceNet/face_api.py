from face_validation import Face_Validation
from face_validation import Face_DB_Operation

import cv2
import os
from api.mtcnn_detect import get_detect_face, get_mtcnn_detector
from PIL import Image

import time

# 保存人脸
def save_face_feature_db(path,name="0"):
    """
        保存人脸
        :param path: 人脸图片路径
        :param name: 人脸名称
    """
    fv = Face_Validation()
    db = Face_DB_Operation()

    img_path = path
    cropped_img_tensor, _ = get_detect_face(img_path, target_size=(128, 128))

    # 保存人脸图片 
    # 将Tensor转换为NumPy数组
    image_array = cropped_img_tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    
    # 使用Pillow保存图像
    image = Image.fromarray(image_array)
    image_name = f'face_image_{name}.png'
    temp_path = os.path.join("temp_face", image_name)
    image.save(temp_path)
    embedding_facture = fv.infer(face_path=temp_path,model=fv.load_model())

    # 保存人脸特征
    db.save_face_feature(face_feature=embedding_facture)

    return embedding_facture


def face_validation(path=None,img=None):
    """
        人脸验证
        :param path: 人脸图片路径
        :param img: 人脸图片数据
    """
    fv = Face_Validation()
    if img is not None:
        cropped_img_tensor, box = get_detect_face(image_data=img, target_size=(128, 128))
    else:
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        cropped_img_tensor, box = get_detect_face(img_path, target_size=(128, 128))
    # res = fv.open_face_validation(face_img=cropped_img_tensor,model=fv.load_model(model_path="model_data/ep001-loss0.493-val_loss0.510.pth"))
    if cropped_img_tensor.ndim > 1:
        print("识别到人脸")
        res = fv.open_face_validation(face_img=cropped_img_tensor,model=fv.load_model())
        return res, box
    else:
        print("未识别到人脸")
        return None, box


if __name__ == "__main__":
    # 保存人脸特征向量
    if False:
        db = Face_DB_Operation()

        # 保存人脸测试
        embedding_facture = save_face_feature_db(path="2_001.jpg", name="test")
        print(f"保存的人脸向量：{embedding_facture}")
        # 加载人脸数据库
        db_v = db.load_vector_db()
        print("向量数据库：")
        print(db_v)

    # 使用摄像头读取图像进行人脸检测:摄像头的图像报-R网络未检测到人脸
    elif True:
        # 建立连接
        cap = cv2.VideoCapture(0)
        num = 0
        while cap.isOpened():

            # 3，读取视频帧
            # 读取一帧，返回：读取状态，图像数据
            status, frame = cap.read()
            num += 1
            if not status:
                print('读取摄像头失败')
                break

            if num % 1 == 0:
                # BGR转RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 使用Pillow保存图像
                image = Image.fromarray(frame_rgb)
                image.save('output_image3.png')

                # 人脸检测并识别
                res, box = face_validation(img=image)
                
                if res:
                    print("人脸验证成功")
                else:
                    print("人脸验证失败")
                if len(box) > 0:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 255, 0),
                                  thickness=5)
                    text = f'{box[4]:.2f}'
                    cv2.putText(frame, text, (int(box[0]), int(box[1]) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 5)
            # 6，显示结果
            cv2.imshow('frame', frame)
            # 按键等待 esc或者q键退出
            if cv2.waitKey(int(1000/24)) == 27 or 0xFF == ord('q'):
                break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
    else:
        # 使用本地图片进行测试人脸检测
        face_validation(path='api/2_001.jpg')