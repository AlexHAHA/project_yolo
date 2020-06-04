import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable

from resnet_yolo import resnet50, resnet18
from yoloLoss import yoloLoss
from dataset import yoloDataset
#from visualize import Visualizer

use_gpu = torch.cuda.is_available()

#file_root     = r'D:\CETCA_DeepLearning\datasets\voc\VOCdevkit\VOC2007\JPEGImages'
file_root     = r'H:\deepLearning\dataset\voc\VOCtrainval_06-Nov-2007\VOC2007\JPEGImages'
learning_rate = 0.001
num_epochs    = 50
batch_size    = 4
use_resnet    = True
if use_resnet:
    net = resnet50()
else:
    pass

print('load pre-trained model')
if use_resnet:
    resnet = models.resnet50()
    resnet.load_state_dict(torch.load(r'H:\deepLearning\weights_cfgs\resnet50-19c8e357.pth'))
    new_state_dict = resnet.state_dict()

    dd = net.state_dict()
    for k in new_state_dict.keys():
        if k in dd.keys() and not k.startswith('fc'):
            print(f'{k} matched')
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
else:
    pass
#load trained weights
if False:
    net.load_state_dict(torch.load('best.pth'))
print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

criterion = yoloLoss(7,2,5,0.5)
if use_gpu:
    net.cuda()
net.train()

#different learning rate
params = []
params_dict = dict(net.named_parameters())
for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params':[value], 'lr':learning_rate*1}]
    else:
        params += [{'params':[value], 'lr':learning_rate}]
#optimizer
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

train_dataset = yoloDataset(root=file_root, list_file=['voc2007train.txt'], train=True, 
                            transform=[transforms.ToTensor()])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
print(f'the dataset has {len(train_dataset)} images')
print(f'the batchsize is {batch_size}')

logfile = open('log.txt', 'w')

num_iter = 0
#vis = Visualizer(env='xiong')
best_test_loss = np.inf 

#train
for epoch in range(num_epochs):
    net.train()
    if epoch == 30:
        learning_rate = 0.0001
    if epoch == 40:
        learning_rate = 0.00001
    # optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate*0.1,momentum=0.9,weight_decay=1e-4)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))

    total_loss = 0.

    for i, (images, target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)

        if use_gpu:
            images, target = images.cuda(), target.cuda()

        pred = net(images)
        loss = criterion(pred, target)
        total_loss += loss.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
                    %(epoch+1, num_epochs, i+1, len(train_loader), loss.data[0], total_loss / (i+1)))
            num_iter += 1
            #vis.plot_train_val(loss_train=total_loss/(i+1))







