import nets
import os
import train

if __name__ == '__main__':
    param_path = r"param1/o_net.pt"
    data_path = r"test_data/MTCNN/48"
    if not os.path.exists("param"):
        os.makedirs("param")
    net = nets.ONet()
    t = train.Trainer(net, param_path, data_path)
    t.train(0.001, landmark=True)
