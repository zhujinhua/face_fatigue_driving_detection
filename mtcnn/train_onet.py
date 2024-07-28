import time

import nets
import os
import train

if __name__ == '__main__':
    param_path = r"param_after/o_net.pt"
    data_path = r"train_data/MTCNN/48"
    if not os.path.exists("param_after"):
        os.makedirs("param_after")
    begin = time.time()
    net = nets.ONet()
    t = train.Trainer(net, param_path, data_path)
    t.train(0.001, landmark=True)
    end = time.time()
    print('ONet training time: %s' % (end - begin))
