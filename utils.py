import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import timedelta
# -i https://pypi.tuna.tsinghua.edu.cn/simple
from PIL import Image
from torchvision import transforms



class Config(object):
    """
    配置参数
    """
    def __init__(self):
        self.os_dir = os.getcwd()

        # self.dataset_name = '/NWPU-RESISC45'
        self.dataset_name = '/AID'
        # self.num_classes = 45
        self.num_classes = 30
        if os.getcwd().find('DataPrep') == -1:
            self.path = self.os_dir + '/DataPrep/data' + self.dataset_name
        else:
            self.path = self.os_dir + '/data' + self.dataset_name
        self.model_name = 'Res2SE50'
        # self.model_name = 'ResNet50'#模型名称
        self.freeze_layers = True
        self.train_path = self.path + '/train-2'   #训练集
        self.valid_path =  self.path + '/valid-8' #验证集
        self.train_per = 0.2
        self.valid_per = 0.8
        self.save_path =  self.path + '/saved_dict/' + self.model_name + '.ckpt'   #模型训练结果
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #设备配置
        self.class2id = {}
        self.id2class = {}
        self.time_run = time.time()
        for line in open( self.path + '/category.txt', encoding='utf-8').readlines():
            line = line.strip('\n')
            id, clas = line.split('--')
            self.class2id.update({clas: int(id)})
            self.id2class.update({int(id): clas})
        self.class_list = [i for i in self.class2id.keys()]
        self.require_improvement = 1000 #若超过1000batch效果还没有提升，提前结束训练
        self.num_epochs = 100#轮次数
        self.batch_size = 32
        self.fc_lr = 0.1 #学习率
        self.se_lr =0.01
        self.sbam_lr =0.01
        self.rest_lr = 0.001
        
        self.time = time.asctime(time.localtime(time.time()))
        self.evaluate_path =  self.path + '/evaluate/' + self.model_name + \
                             '_Epoch' + str(self.num_epochs) + \
                             '_Batch' + str(self.batch_size) + \
                              'lr'    + str(self.fc_lr)+str(self.se_lr)+str(self.sbam_lr)+str(self.rest_lr)+ self.time +'.csv'  # 验证效果
        self.process_path =  self.path + '/process/' + self.model_name + \
                             '_Epoch' + str(self.num_epochs) + \
                             '_Batch' + str(self.batch_size) +'lr'+ str(self.fc_lr)+str(self.se_lr)+str(self.sbam_lr)+str(self.rest_lr)+ self.time +'.csv'  # 验证效果



def get_time_dif(start_time):
    """
    获取使用时间
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def load_predict_data(config, file_path):
    """
    加载预测数据
    """
    img = Image.open(file_path)

    pipline_predict = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    ])
    img_trans = pipline_predict(img)
    image = img_trans.unsqueeze(0)
    return image

if __name__ == '__main__':
    # img_tensor = transforms.ToTensor()(img)  # tensor数据格式是torch(C,H,W)
    pass