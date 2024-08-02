from PIL import Image

# 网络结构
from nets.facenet import Facenet as facenet
from torchvision import transforms
import torch
import numpy as np
import os

# 解决OMP问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Face_Validation():
    def __init__(self) -> None:
        pass

    # 数据预处理
    def data_preprocess_infer(self, data):
        """
            数据预处理： 单张图像
            :param data: 待处理的数据
        """
        if data is None:
            raise ValueError("data is None")

        if not isinstance(data, torch.Tensor):
            data = np.array(data)
            data = torch.tensor(data=data, dtype=torch.float32)
            # 转 维度 [H, W, C] ---> [C, H, W]
            data = data.permute(dims=(2, 0, 1))

        # 中心裁剪
        # if isinstance(data, torch.Tensor) :
        data = transforms.CenterCrop(size=max(data.shape))(data)
        # elif isinstance(data, np.ndarray):
        #     data = torch.tensor(data = data,dtype=torch.float32)
        #     data = transforms.CenterCrop(size=max(data.shape))(data)
        # else: 
        #     data = transforms.CenterCrop(size=max(data.size))(data)
        data = transforms.Resize(size=(160, 160))(data)

        # if not isinstance(data, torch.Tensor):
        #     data = np.array(data)
        #     data = torch.tensor(data = data,dtype=torch.float32)

        # 预处理 ：[-1,1]
        data = ((data / 255) - 0.5) / 0.5

        # 添加batch 维度
        data = data.unsqueeze(dim=0)
        return data

    # 模型加载
    def load_model(self, model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data", 'facenet_mobilenet.pth'), backbone="mobilenet"):
        """
            加载模型
            :param model_path: 模型路径
            :param backbone: 模型类型
            :return: 模型对象
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = facenet(backbone=backbone, mode="predict").eval()
        net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print('{} model loaded.'.format(model_path))
        return net

    # 模型推理API
    def infer(self, face_img=None, model=None, face_path=None, is_cuda=False):
        """
            模型推理API
            :param face_img: 待预测的人脸图片
            :param model: 待预测的模型
            :param face_path: 待预测的人脸图片路径
            :return: 预测结果 128维特征向量
        """
        if face_img is None and face_path is None:
            # print("face_img and face_path can not be None at the same time!")
            raise ValueError("face_img and face_path can not be None at the same time!")

        if face_img is not None:
            # 输入图像数据预处理
            # img = self.data_preprocess_infer(face_img)
            # print(img.shape)
            img = face_img
            img = img.unsqueeze(dim=0)

        if face_path is not None:
            # 读取待预测的人脸图片
            img = Image.open(face_path)
            # 输入图像数据预处理
            img = self.data_preprocess_infer(img)

        # 模型推理
        with torch.no_grad():
            if is_cuda:
                img = img.cuda()
                model = model.cuda()
            feature = model(img)
        return feature

    # 人脸验证
    def open_face_validation(self, face_img=None, model=None, face_path=None, is_cuda=False, threshold=1.1):
        """
            人脸验证
            :param face_img: 待预测的人脸图片
            :param model: 待预测的模型
            :param face_path: 待预测的人脸图片路径
            :param is_cuda: 是否使用GPU
            :param threshold: 阈值
            :return: 人脸验证结果 0: 验证失败 1: 验证成功
        """
        # 推理
        feature = self.infer(face_img=face_img, model=model, face_path=face_path, is_cuda=is_cuda)

        # 加载向量数据库 
        db = Face_DB_Operation()
        vector_db = db.load_vector_db()

        if vector_db is None or len(vector_db) == 0:
            raise ValueError("vector database is None")

        # 人脸验证
        for index, vector in enumerate(vector_db):
            print(np.linalg.norm((feature.cpu() - vector).numpy()))
            if np.linalg.norm((feature.cpu() - vector).numpy()) < threshold:
                return 1

        return 0


class Face_DB_Operation():
    def __init__(self) -> None:
        pass

    # 加载向量数据库
    def load_vector_db(self, db_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_db", 'face_db.bin')):
        """
            加载向量数据库
            :Param db_path: str, 默认 "./vector_db/face_db.bin"
        """
        if os.path.exists(db_path):
            return torch.load(f=db_path)
        return None

    # 保存人脸特征向量至数据库中
    def save_face_feature(self, face_feature, name=None, db_path="./vector_db/face_db.bin"):

        """
            保存人脸特征向量
            :param face_feature: 人脸特征向量
            :param name: 人脸名称
            :param db_path: 数据库路径
        """
        face_feature_dict = {}
        if os.path.exists(db_path):
            face_feature_dict = torch.load(f=db_path)

        if name is None:
            name = str(len(face_feature_dict) + 1)

        face_feature_dict[face_feature] = name

        torch.save(face_feature_dict, db_path)
        print(f"{name}特征向量保存成功")


if __name__ == "__main__":
    fv = Face_Validation()
    db = Face_DB_Operation()
    # 提取人脸特征
    embedding_facture = fv.infer(face_path="output_image3.png", model=fv.load_model())

    # 保存人脸特征
    db.save_face_feature(face_feature=embedding_facture)
    # # 加载向量库
    # db_v = db.load_vector_db()
    # # print(db_v)

    # # 人脸验证
    # # open_face_validation(face_path="img/1_001.jpg",model=load_model())
    # res = fv.open_face_validation(face_path="output_image2.png",model=fv.load_model())
    # print(f"测试2_001.jpg：{res}")

    # 人脸检测
    import os

    from api.mtcnn_detect import get_detect_face

    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api', 'Aaron_Sorkin_0001.jpg')
    cropped_img_tensor, _ = get_detect_face(img_path, target_size=(128, 128))
    # [channels, height, weight]
    # print(cropped_img_tensor.shape)

    # 保存人脸图片 
    # # 将Tensor转换为NumPy数组
    # image_array = cropped_img_tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

    # # 使用Pillow保存图像
    # image = Image.fromarray(image_array)
    # image.save('output_image3.png')

    res = fv.open_face_validation(face_img=cropped_img_tensor, model=fv.load_model())
    print(res)
