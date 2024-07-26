import nets
import os
import train

if __name__ == '__main__':
    param_path = r"param_before/r_net.pt"
    data_path = r"train_data/MTCNN/24"
    if not os.path.exists("param_before"):
        os.makedirs("param_before")
    net = nets.RNet()
    t = train.Trainer(net, param_path, data_path)
    t.train(0.001)
