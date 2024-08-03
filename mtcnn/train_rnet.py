import time

import nets
import os
import train

if __name__ == '__main__':
    param_path = r"param/r_net.pt"
    train_data_path = r"train_data/MTCNN/24"
    test_data_path = r"test_data/MTCNN/24"
    if not os.path.exists("param"):
        os.makedirs("param")
    net = nets.RNet()
    begin = time.time()
    t = train.Trainer(net, param_path, train_data_path, test_data_path)
    t.train(0.001)
    end = time.time()
    print('RNet training time %s' % (end - begin))
