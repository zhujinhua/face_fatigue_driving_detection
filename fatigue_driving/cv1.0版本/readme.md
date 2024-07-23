## 本地试运行程序步骤（此版本仅为了先把程序在本地跑起来）
### 1、settings文件路径修改
C:\Users\Lenovo\AppData\Roaming\Ultralytics\settings.yaml文件配置修改datasets_dir
例如：datasets_dir: D:\AI50期直通车\yolo_materials\datasets\FatigueTest
### 2、配置修改过的数据集FatigueTest
1、该数据集只截取十张图片，为了程序能快速跑完
2、原来的图片文件夹JPEGImages改成images，否则yolo不认识报警告
3、FatigueTest放在yolo的专用文件夹中，例如：D:\AI50期直通车\yolo_materials\datasets
### 3、修改程序中的dataset_path为FatigueTest路径
例如：dataset_path = 'D:/AI50期直通车/yolo_materials/datasets/FatigueTest/'
### 4、运行程序
### 5、结果查看
在yolo的专用文件夹的detect中查看结果
例如:D:\AI50期直通车\yolo_materials\ultralytics\runs\detect\train22