
'''
txt描述文件 image_name.jpg x y w h c x y w h c 这样就是说一张图片中有两个目标
'''
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
            # 如果list_file是包含多个file的list，则将多个file的内容进行合并
            # linux下的文件合并方法
            #tmp_file = '/tmp/listfile.txt'
            #os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            #list_file = tmp_file
            
            # Windows下的文件合并方法
            tmp_file = open('tmp.txt','w')
            for fi in list_file:
                with open(fi) as f:
                    cont = f.readlines()
                    tmp_file.writelines(cont)
                    tmp_file.write('\n')
            tmp_file.close()
            list_file = 'tmp.txt'
            
        with open(list_file) as f:
            lines = f.readlines()
        
        for line in lines:
            if line == '\n':
                #print(f'dataset.py: line={line}')
                continue
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
        #img    = cv2.imread(os.path.join(self.root+fname))
        #print(f'dataset.py:{os.path.join(self.root, fname)}')
        img    = cv2.imread(os.path.join(self.root, fname))
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
        #print(f'dataset.py-encoder: boxes={boxes}')
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
            #print(f'{target[int(ij[1]), int(ij[0]), 2:4]}')
            #print(f'{wh[i]}')
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target
    
    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    def RandomBrightness(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomHue(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self,bgr):
        if random.random()<0.5:
            bgr = cv2.blur(bgr,(5,5))
        return bgr

    def randomShift(self,bgr,boxes,labels):
        #平移变换
        center = (boxes[:,2:]+boxes[:,:2])/2
        if random.random() <0.5:
            height,width,c = bgr.shape
            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
            after_shfit_image[:,:,:] = (104,117,123) #bgr
            shift_x = random.uniform(-width*0.2,width*0.2)
            shift_y = random.uniform(-height*0.2,height*0.2)
            #print(bgr.shape,shift_x,shift_y)
            #原图像的平移
            if shift_x>=0 and shift_y>=0:
                after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x>=0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]

            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width)
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image,boxes_in,labels_in
        return bgr,boxes,labels

    def randomScale(self,bgr,boxes):
        #固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8,1.2)
            height,width,c = bgr.shape
            bgr = cv2.resize(bgr,(int(width*scale),height))
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr,boxes
        return bgr,boxes

    def randomCrop(self,bgr,boxes,labels):
        if random.random() < 0.5:
            center = (boxes[:,2:]+boxes[:,:2])/2
            height,width,c = bgr.shape
            h = random.uniform(0.6*height,height)
            w = random.uniform(0.6*width,width)
            x = random.uniform(0,width-w)
            y = random.uniform(0,height-h)
            x,y,h,w = int(x),int(y),int(h),int(w)

            center = center - torch.FloatTensor([[x,y]]).expand_as(center)
            mask1 = (center[:,0]>0) & (center[:,0]<w)
            mask2 = (center[:,1]>0) & (center[:,1]<h)
            mask = (mask1 & mask2).view(-1,1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if(len(boxes_in)==0):
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:,0]=boxes_in[:,0].clamp_(min=0,max=w)
            boxes_in[:,2]=boxes_in[:,2].clamp_(min=0,max=w)
            boxes_in[:,1]=boxes_in[:,1].clamp_(min=0,max=h)
            boxes_in[:,3]=boxes_in[:,3].clamp_(min=0,max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y+h,x:x+w,:]
            return img_croped,boxes_in,labels_in
        return bgr,boxes,labels

    def subMean(self,bgr,mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes
    
    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta,delta)
            im = im.clip(min=0,max=255).astype(np.uint8)
        return im

def main():
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    file_root = r'D:\CETCA_DeepLearning\datasets\voc\VOCdevkit\VOC2007\JPEGImages'
    train_dataset = yoloDataset(root=file_root,list_file='voc2007train.txt',train=True,transform = [transforms.ToTensor()] )
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0)
    train_iter = iter(train_loader)
    
    for i in range(3):
        print(f'idx={i}')
        img,target = next(train_iter)
        #print(img,target)
        print(f'image shape:{img.shape}')
        
        mask = target[0,:,:,4]>0
        mask = mask.unsqueeze(-1).expand_as(target)
        result = target[mask].view(-1,30)
        print(f'result shape:{result.shape}')
        print(result)

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    