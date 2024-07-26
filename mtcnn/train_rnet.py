import time

import nets
import os
import train

if __name__ == '__main__':
    param_path = r"param/r_net.pt"
    data_path = r"train_data/MTCNN/24"
    if not os.path.exists("param"):
        os.makedirs("param")
    begin = time.time()
    net = nets.RNet()
    t = train.Trainer(net, param_path, data_path)
    t.train(0.001)
    end = time.time()
    print('RNet training cost: %s' % (end - begin))
