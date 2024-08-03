import time

import nets
import os
import train

if __name__ == '__main__':
    param_path = r"param/o_net.pt"
    train_data_path = r"train_data/MTCNN/48"
    test_data_path = r"test_data/MTCNN/48"
    if not os.path.exists("param"):
        os.makedirs("param")
    net = nets.ONet()
    begin = time.time()
    t = train.Trainer(net, param_path, train_data_path, test_data_path)
    t.train(0.001, landmark=True)
    end = time.time()
    print('ONet training time %s' % (end - begin))
