# FileName:rename_dic.py
# Author:Li Rumei
# Time :2022/10/28 12:54

#将文件夹名称改为自然序号，生成一个文件夹对应类别的txt文件
import os

import utils


def rename():
    i = -1
    config = utils.Config()
    path = path = config.path + '/' + config.dataset_name

    filelist = os.listdir(path)   #该文件夹下所有的文件（包括文件夹）

    def save_txt(str_list: list, name):
        j=-1
        with open(name, 'w', encoding='utf-8') as f:
            for i in str_list:
                j = j + 1
                f.write(str(j)+ '--' + i + '\n')

    save_txt(filelist,'category.txt')
    for files in filelist:   #遍历所有文件
        i = i + 1
        Olddir = os.path.join(path, files)    #原来的文件路径
        if os.path.isdir(os.path.join(path,files))==False:       #如果是文件则跳过
                continue
        Newdir = os.path.join(path, str(i))   #新的文件路径
        os.rename(Olddir, Newdir)    #重命名
    return True

if __name__ == '__main__':
    rename()
