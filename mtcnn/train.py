import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data.face_dataset import FaceDataset


class Trainer:
    def __init__(self, net, param_path, train_data_path, test_data_path):
        self.device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        self.datasets = FaceDataset(train_data_path)
        self.test_dataset = FaceDataset(test_data_path)
        self.net = net.to(self.device)
        self.param_path = param_path
        self.cls_loss_func = torch.nn.BCELoss()
        self.offset_loss_func = torch.nn.MSELoss()
        self.point_loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=1e-3)

    def evaluate(self, dataloader, landmark=False):
        self.net.eval()
        total_loss = 0
        with torch.no_grad():
            for i, (img_data, _cls, _offset, _point) in enumerate(dataloader):
                img_data = img_data.to(self.device)
                _cls = _cls.to(self.device)
                _offset = _offset.to(self.device)
                _point = _point.to(self.device)

                if landmark:
                    out_cls, out_offset, out_point = self.net(img_data)
                else:
                    out_cls, out_offset = self.net(img_data)

                out_cls = out_cls.view(-1, 1)
                out_offset = out_offset.view(-1, 4)

                cls_mask = torch.lt(_cls, 2)
                cls = torch.masked_select(_cls, cls_mask)
                out_cls = torch.masked_select(out_cls, cls_mask)
                cls_loss = self.cls_loss_func(out_cls, cls)

                offset_mask = torch.gt(_cls, 0)
                offset = torch.masked_select(_offset, offset_mask)
                out_offset = torch.masked_select(out_offset, offset_mask)
                offset_loss = self.offset_loss_func(out_offset, offset)

                if landmark:
                    point = torch.masked_select(_point, offset_mask)
                    out_point = torch.masked_select(out_point, offset_mask)
                    point_loss = self.point_loss_func(out_point, point)
                    loss = cls_loss + offset_loss + point_loss
                else:
                    loss = cls_loss + offset_loss

                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        self.net.train()
        return avg_loss

    def train(self, stop_value, landmark=False):
        if os.path.exists(self.param_path):
            self.net.load_state_dict(torch.load(self.param_path, map_location=self.device), strict=False)
            print("加载参数文件，继续训练 ...")
        else:
            print("没有参数文件，全新训练 ...")

        dataloader = DataLoader(self.datasets, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        epochs = 0
        all_train_losses = []
        all_test_losses = []

        while True:
            epoch_losses = []
            for i, (img_data, _cls, _offset, _point) in enumerate(dataloader):
                img_data = img_data.to(self.device)
                _cls = _cls.to(self.device)
                _offset = _offset.to(self.device)
                _point = _point.to(self.device)

                if landmark:
                    out_cls, out_offset, out_point = self.net(img_data)
                else:
                    out_cls, out_offset = self.net(img_data)

                out_cls = out_cls.view(-1, 1)
                out_offset = out_offset.view(-1, 4)

                cls_mask = torch.lt(_cls, 2)
                cls = torch.masked_select(_cls, cls_mask)
                out_cls = torch.masked_select(out_cls, cls_mask)
                cls_loss = self.cls_loss_func(out_cls, cls)

                offset_mask = torch.gt(_cls, 0)
                offset = torch.masked_select(_offset, offset_mask)
                out_offset = torch.masked_select(out_offset, offset_mask)
                offset_loss = self.offset_loss_func(out_offset, offset)

                if landmark:
                    point = torch.masked_select(_point, offset_mask)
                    out_point = torch.masked_select(out_point, offset_mask)
                    point_loss = self.point_loss_func(out_point, point)
                    loss = cls_loss + offset_loss + point_loss
                else:
                    loss = cls_loss + offset_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            all_train_losses.append(avg_train_loss)
            avg_test_loss = self.evaluate(test_dataloader, landmark)
            all_test_losses.append(avg_test_loss)

            print(f"epoch: {epochs}, train_loss: {avg_train_loss:.4f}, test_loss: {avg_test_loss:.4f}")
            torch.save(self.net.state_dict(), self.param_path)

            epochs += 1

            if avg_train_loss < stop_value:
                break

            plt.figure()
            plt.plot(range(len(all_train_losses)), all_train_losses, label='Train Loss')
            plt.plot(range(len(all_test_losses)), all_test_losses, label='Test Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('%s Training and Testing Loss Curve' % str.upper(self.param_path.split('/')[-1].split('.')[0]))
            plt.legend()
            plt.show()
