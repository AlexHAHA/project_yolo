"""
    imagenet classification model
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

class MyNet(nn.Module):
    """
    Used as imageNet classifier

    conv layer:
    -input size: (N, 3, 224, 224)
    -out size: (N, 1024, 7, 7)
    avgpool layer:
    -out size: (N, 1024)
    fc layer:
    -out size: (N, 1000)
    """
    def __init__(self):
        super(MyNet,self).__init__()

        self.layer1 = nn.Sequential(OrderedDict([
                        ('conv1', nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)),
                        ('bn1', nn.BatchNorm2d(64)),
                        ('relu1', nn.ReLU(inplace=True)),
                        ('maxpool1', nn.MaxPool2d(kernel_size=2,stride=2))
                    ]))
        self.layer2 = nn.Sequential(OrderedDict([
                        ('conv1', nn.Conv2d(64,192,kernel_size=3,stride=1,padding=1)),
                        ('bn1', nn.BatchNorm2d(192)),
                        ('relu1', nn.ReLU(inplace=True)),
                        ('maxpool1', nn.MaxPool2d(kernel_size=2,stride=2))
                    ]))
        self.layer3 = nn.Sequential(OrderedDict([
                        ('conv1', nn.Conv2d(192,128,kernel_size=1)),
                        ('bn1', nn.BatchNorm2d(128)),
                        ('relu1', nn.ReLU(inplace=True)),
                        ('conv2', nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1)),
                        ('bn2', nn.BatchNorm2d(256)),
                        ('relu2', nn.ReLU(inplace=True)),
                        ('conv3', nn.Conv2d(256,512,kernel_size=1)),
                        ('bn3', nn.BatchNorm2d(512)),
                        ('relu3', nn.ReLU(inplace=True)),
                        ('maxpool1', nn.MaxPool2d(kernel_size=2,stride=2))
                    ]))
        self.layer4 = nn.Sequential(OrderedDict([
                        ('conv1', nn.Conv2d(512,256,kernel_size=1)),
                        ('bn1', nn.BatchNorm2d(256)),
                        ('relu1', nn.ReLU(inplace=True)),
                        ('conv2', nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1)),
                        ('bn2', nn.BatchNorm2d(512)),
                        ('relu2', nn.ReLU(inplace=True)),
                        ('conv3', nn.Conv2d(512,1024,kernel_size=1)),
                        ('bn3', nn.BatchNorm2d(1024)),
                        ('relu3', nn.ReLU(inplace=True)),
                        ('maxpool1', nn.MaxPool2d(kernel_size=2,stride=2))
                    ]))
        self.layer5 = nn.Sequential(OrderedDict([
                        ('conv1', nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=1)),
                        ('bn1', nn.BatchNorm2d(1024)),
                        ('relu1', nn.ReLU(inplace=True))
                    ]))
        # without avgpool
        #self.fc1   = nn.Linear(1024*7*7,4096)
        # with avgpool
        self.avgpool = nn.AvgPool2d(7)        #fit 224 input size
        self.fc      = nn.Linear(1024,1000)

    def forward(self, x):
        x = self.layer1(x)
        #print(f'layer1:{x.shape}')
        x = self.layer2(x)
        #print(f'layer2:{x.shape}')
        x = self.layer3(x)
        #print(f'layer3:{x.shape}')
        x = self.layer4(x)
        #print(f'layer4:{x.shape}')
        x = self.layer5(x)
        #print(f'layer5:{x.shape}')
        x = self.avgpool(x)
        #print(f'avgpool:{x.shape}')

        x = x.view(x.size(0), -1)
        out = F.relu(self.fc(x))
        #print(f'fc:{out.shape}')
        return out


def train():
    epochs     = 100
    batch_size = 8
    net = MyNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            inputs, targets = data
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

if __name__ == '__main__':
    pass