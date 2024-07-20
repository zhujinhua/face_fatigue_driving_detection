import nets
import os
import train

if __name__ == '__main__':
    # 权重存放地址
    param_path = r"param1/p_net.pt"
    # 数据存放地址
    data_path = r"test_data/MTCNN/12"
    
    # 如果没有这个参数存放目录，则创建一个目录
    if not os.path.exists("param1"):
        os.makedirs("param1")

    # 构建模型
    pnet = nets.PNet()

    # 开始训练
    t = train.Trainer(pnet, param_path, data_path)

    t.train(0.01)
