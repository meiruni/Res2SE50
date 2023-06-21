import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import sys
from PIL import Image
import random
import torch
import torchvision.transforms.functional as F
sys.path.append("..")
import utils
config = utils.Config()
dataset = config.path

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

def gen_txt(txt_path, img_dir):
    f = open(txt_path, 'w')

    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)  # 获取各类的文件夹 绝对路径
            img_list = os.listdir(i_dir)  # 获取类别文件夹下所有jpg图片的路径
            for i in range(len(img_list)):
                if not img_list[i].endswith('jpg'):  # 若不是jpg文件，跳过
                    continue
                label = img_list[i].split('_')[0]
                img_path = os.path.join(i_dir, img_list[i])
                line = img_path + ' ' + sub_dir + '\n'
                f.write(line)
    f.close()

pipline_train = transforms.Compose([
        # transforms.Resize(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
         # transforms.Resize(256),
         transforms.CenterCrop(224),  # 从中心开始裁剪
         transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
         transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
 
])


pipline_test = transforms.Compose([
        # transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


gen_txt(config.train_path + '.txt', config.train_path)
gen_txt(config.valid_path + '.txt', config.valid_path)

train_data = MyDataset(config.train_path + '.txt', transform=pipline_train)
test_data = MyDataset(config.valid_path + '.txt', transform=pipline_test)
# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)

# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


#train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=True)




