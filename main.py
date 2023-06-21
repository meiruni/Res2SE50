
from utils import Config
import train
import test
from model import *
from DataPrep.DataLoder import testloader,trainloader

if __name__ == '__main__':
    config = Config()
    model = Res2SE50(config.num_classes).to(config.device) #模型对象
    train.train(config, model,trainloader, testloader) #训练模型
    test.test(config, model, testloader) #测试模型效果