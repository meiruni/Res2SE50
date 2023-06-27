import math

import torch
import torch.nn as nn
import torchvision
import torch.nn as nn
import torchvision.models as models




def VGG16(num_classes):
    #VGG16网络
    VGG16 = torchvision.models.vgg16(pretrained = True)
    VGG16.classifier[6] = nn.Linear(4096,num_classes)
    return VGG16

def resnet50(num_classes):
    #ResNet50网络
    resnet50 = torchvision.models.resnet50(pretrained=True)
    resnet50.fc = nn.Linear(2048,num_classes)
    return resnet50

#densenet121网络
def densenet121(num_classes):

    densenet121 = torchvision.models.densenet121(pretrained = True)
    densenet121.fc =nn.Linear(4096,num_classes)
    return densenet121






# 注意力机制模块
class DSE(nn.Module):
    def __init__(self, in_channel, reduction):
        super(DSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x_se = self.avg_pool(x).view(b, c, 1, 1)
        x_se = self.fc(x_se)
        return x * x_se.expand_as(x)

class CSE(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(CSE, self).__init__()
        self.channel_squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channel // reduction),
            nn.ReLU(inplace=True)
        )
        self.spatial_excitation = nn.Sequential(
            nn.Conv2d(in_channel // reduction, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.channel_squeeze(x)
        y = self.spatial_excitation(y)
        return x * y


class Res2SE50(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Res2SE50, self).__init__()

        self.num_classes = num_classes
        resnet = models.resnet50(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1, DSE(256, reduction=16), CSE(256, reduction=16))
        self.layer2 = nn.Sequential(resnet.layer2, DSE(512, reduction=16), CSE(512, reduction=16))
        self.layer3 = nn.Sequential(resnet.layer3, DSE(1024, reduction=16), CSE(1024, reduction=16))
        self.layer4 = nn.Sequential(resnet.layer4, DSE(2048, reduction=16), CSE(2048, reduction=16))
        self.avgpool = resnet.avgpool

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

