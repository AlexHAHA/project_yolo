import os
import sys
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt

class yoloDataset(data.Dataset):
    image_size = 448
    def __init__(self, root, list_file, train, transform):
        self.root  = root
        self.train = train
        self.transform = transform
        self.fnames = []
        self.boxes  = []
        self.labels = []
        self.mean = (123, 117, 104) #RGB
        
        if isinstance(list_file, list):
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file
            
        with open(list_file) as f:
            lines = f.readlines()
        
        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                x = float(splited[1+5*i])
                y = float(splited[2+5*i])
                x2 = float(splited[3+5*i])
                y2 = float(splited[4+5*i])
                c = splited[5+5*i]
                box.append([x,y,x2,y2])
                label.append(int(c)+1)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)
        
    def __getitem__(self, idx):
        fname  = self.fnames[idx]
        img    = cv2.imread(os.path.join(self.root+fname))
        boxes  = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        '''
        if self.train:
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.randomScale(img, boxes)
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)
            img, boxes, labels = self.randomShift(img, boxes, labels)
            img, boxes, labels = self.randomCrop(img, boxes, labels)
        '''
        h, w, _ = img.shape
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        img = self.BGR2RGB(img)
        img = self.subMean(img, self.mean)
        img = cv2.resize(img, (self.image_size, self.image_size))
        target = self.encoder(boxes, labels) #7x7x30
        for t in self.transform:
            img = t(img)
        return img, target
    
    def __len__(self):
        return self.num_samples
    
    def encoder(self, boxes, labels):
        '''
        Args:
        - boxes: tensor,[[x1,y1,x2,y2],[]]
                x1,y1,x2,y2都是相对于pic大小的归一化的值,即认为pic_size为单位1
        - labels: tensor, [....]
        Return:
        - target: tensor of shape 7x7x30, 
                  channels:[box1,box2,prob]=[x,y,w,h,conf,x,y,w,h,conf,p(0),p(1),..,p(c-1)]
            其中x,y是box中心点相对于grid左上角坐标的偏移量，并且归一化后的值（以grid大小为单位1进行归一化）
            其中w,h是box宽和高相对于pic大小的归一化的值,,即认为pic_size为单位1
        '''
        grid_num = 7
        target = torch.zeros((grid_num, grid_num, 30))
        # cell_size也是相对于pic大小归一化后的值
        cell_size = 1./grid_num
        
        # width=x2-x and height=y2-y
        wh = boxes[:,2:] - boxes[:,:2]
        # grid center:cx=(x2+x)/2 , cy=(y2+y)/2
        cxcy = (boxes[:,2:] + boxes[:,:2]) / 2
        
        # 根据每个box所在的grid，设置对应grid
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample / cell_size).ceil() - 1
            # box1's confidence
            target[int(ij[1]), int(ij[0]), 4] = 1
            # box2's confidence
            target[int(ij[1]), int(ij[0]), 9] = 1
            # classes probility
            target[int(ij[1]), int(ij[0]), int(labels[i])+9] = 1
            #
            xy = ij*cell_size
            delta_xy = (cxcy_sample - xy) / cell_size
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:8] = wh[i]
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target
    
    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    
    
    
    
    
    
    
    
    
    
    
    
    