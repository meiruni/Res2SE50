"""
    将原始数据集进行划分成训练集、验证集和测试集
"""
import os
import glob
import random
import shutil
import sys
sys.path.append("..")
import utils

config = utils.Config()
dataset_dir = config.path + '/' + config.dataset_name 
train_dir = config.train_path + '/'
valid_dir = config.valid_path + '/'
test_dir = './data/test/'

train_per = config.train_per
valid_per = config.valid_per

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def cal_weight(ll: list) -> list:
    ll = [sum(ll) / x for x in ll]
    ll = [round(x / max(ll), 3) for x in ll]
    return ll


if __name__ == '__main__':
    class_weight = []
    for root, dirs, files in os.walk(dataset_dir):
        for sDir in dirs:
            imgs_list = glob.glob(os.path.join(root, sDir) + '/*.jpg')
            random.seed()
            random.shuffle(imgs_list)
            imgs_num = len(imgs_list)
            class_weight.append(imgs_num)

            train_point = int(imgs_num * train_per)
            valid_point = int(imgs_num * (train_per + valid_per))

            for i in range(imgs_num):
                if i < train_point:
                    out_dir = train_dir + sDir + '/'
                elif i < valid_point:
                    out_dir = valid_dir + sDir + '/'
                else:
                    out_dir = test_dir + sDir + '/'

                makedir(out_dir)
                out_path = out_dir + os.path.split(imgs_list[i])[-1]
                shutil.copy(imgs_list[i], out_path)

            print('Class:{}, train:{}, valid:{}'.format(sDir, train_point, valid_point - train_point,
                                                            ))
    print(cal_weight(class_weight))
