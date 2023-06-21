
import time

import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from sklearn import metrics

from model import *
from utils import get_time_dif
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
loss_fct = nn.CrossEntropyLoss()




def train(config, model, trainloader, testloader):
    """
    模型训练方法
    """
    loss_all = np.array([], dtype=float)
    label_all = np.array([], dtype=float)
    predict_all = np.array([], dtype=float)
    #创建列表保存train_loss、train_acc
    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []
    start_time = time.time()

    se_params = []
    sbam_params = []
    for name, module in model.named_modules():
        if isinstance(module, DSE):
            for param_name, param in module.named_parameters():
                se_params.append(param)
        elif isinstance(module, CSE):
            for param_name, param in module.named_parameters():
                sbam_params.append(param)
    fc_params = list(model.fc.parameters())
    rest_params = list(
        filter(lambda x: id(x) not in set(map(id, se_params + sbam_params + fc_params)), model.parameters()))
    optimizer = torch.optim.SGD([
        {'params': fc_params, 'lr': config.fc_lr},
        {'params': se_params, 'lr': config.se_lr},
        {'params': sbam_params, 'lr': config.sbam_lr},
        {'params': rest_params, 'lr': config.rest_lr}
    ],momentum=0.9,weight_decay=0.0005,nesterov=True)
    
    
    # # 模型backbone对比
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=0.001,
    #     momentum=0.9,
    #     weight_decay=0.0005,
    #     nesterov=True
    # )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 设定优优化器更新的时刻表
 
    dev_best_loss = float('inf')
    dev_best_f1 = float('-inf')

    for epoch in range(1, config.num_epochs + 1):
        train_loss = 0.
        train_right, train_total = 0, 0
        # train_bar = tqdm(trainloader)
        scheduler.step()
        for batch_idx, batch_data in enumerate(trainloader):
            model.train()
            imgs, labels = batch_data
            optimizer.zero_grad()
            predicts = model(imgs.to(config.device))
            loss = loss_fct(predicts, labels.to(config.device))
            predicts = torch.argmax(predicts, 1)
            loss_all = np.append(loss_all, loss.data.item())
            label_all = np.append(label_all, labels.data.cpu().numpy())
            predict_all = np.append(predict_all, predicts.data.cpu().numpy())
            acc = metrics.accuracy_score(predict_all, label_all)


            loss.backward()
            optimizer.step()
            model.zero_grad()
            if batch_idx % 20 == 0:
                time_dif = get_time_dif(start_time)
                print("Epoch:{}--------Iter:{}--------Time:{}--------train_loss:{:.3f}--------train_acc:{:.3f}"
                      .format(epoch, batch_idx + 1, time_dif,  loss_all.mean(), acc))
        train_acc_list.append(acc)
        train_loss_list.append(loss_all.mean())


        test_loss, test_acc, test_f1, test_report, test_confusion = evaluate(model, testloader, config)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

        msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}, Test F1 :{2:>6.2%}'
        print(msg.format(test_loss, test_acc, test_f1))
        print("Precision, Recall and F1-Score")
        print(test_report)
        print("Test Confusion")
        print(test_confusion)

        if test_f1 >= dev_best_f1:
            dev_best_f1 = test_f1
            torch.save(model.state_dict(), config.save_path)
            improve = '* * * * * * * * * * * * * * Save Model * * * * * * * * * * * * * *'
            print(improve)

        # 将数据保存在.csv文件中
        name = ['train_loss_all', 'test_loss_all', 'train_accur_all', 'test_accur_all']
        list_data= []
        list_data.append(train_loss_list)
        list_data.append(test_loss_list)
        list_data.append(train_acc_list)
        list_data.append(test_acc_list)
        Dname = pd.DataFrame(index=name, data=list_data)
        Dname = Dname.T
        Dname.to_csv(config.process_path, encoding='utf-8')


def evaluate(model, testloader, config, output_dict=False):
    """
    验证模型效果
    """
    model.eval()    #评价模式
    loss_all = np.array([], dtype=float)
    predict_all = np.array([], dtype=int)
    label_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testloader):
            imgs, labels = batch_data
            label_predict = model(imgs.to(config.device))
            loss = loss_fct(label_predict,labels.to(config.device))
            label_predict = torch.argmax(label_predict,1)
            loss_all = np.append(loss_all, loss.data.item())
            predict_all = np.append(predict_all, label_predict.data.cpu().numpy())
            label_all = np.append(label_all, labels.data.cpu().numpy())
    acc = metrics.accuracy_score(label_all, predict_all)
    f1 = metrics.f1_score(label_all, predict_all, average='macro')
    report = metrics.classification_report(label_all, predict_all, target_names=config.class_list, digits=3, output_dict=output_dict)
    confusion = metrics.confusion_matrix(label_all, predict_all)

    return loss_all.mean(), acc, f1, report, confusion
