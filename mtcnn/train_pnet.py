import time

import nets
import os
import train

if __name__ == '__main__':
    # 权重存放地址
    param_path = r"param/p_net.pt"
    # 数据存放地址
    data_path = r"test_data/MTCNN/12"
    
    # 如果没有这个参数存放目录，则创建一个目录
    if not os.path.exists("param"):
        os.makedirs("param")

    # 构建模型
    pnet = nets.PNet()
    # 开始训练
    begin = time.time()
    t = train.Trainer(pnet, param_path, data_path)
    t.train(0.01)
    end = time.time()
    print('PNet training cost: %s' % (end - begin))
