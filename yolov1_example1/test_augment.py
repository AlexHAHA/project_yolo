import numpy as np
import random
import cv2

## HSV增强
def RandomBrightness(bgr):
    '''
    图像亮度调整
    1.随机判断是否进行亮度调整
    2.随机选择变明亮或者变暗
    '''
    if random.random() < 0.5:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        #在列表中随机选择一个元素作为返回值
        adjust = random.choice([0.5,1.5])
        print(f'RandomBrightness:{adjust}')
        v = v*adjust
        #np.clip将array v中的元素限制在[0,255]区间
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h,s,v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr
def RandomSaturation(bgr):
    if random.random() < 0.5:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        adjust = random.choice([0.5,1.5])
        s = s*adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h,s,v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr
def RandomHue(bgr):
    if random.random() < 0.5:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        adjust = random.choice([0.5,1.5])
        h = h*adjust
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h,s,v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def test_hsv(type='v'):
    img = cv2.imread('dog.jpg')
    cv2.imshow('original', img)
    if type=='v':
        img_brightness = RandomBrightness(img)
        cv2.imshow('brightness', img_brightness)    
    elif type=='s':
        img_saturation = RandomSaturation(img)
        cv2.imshow('saturation', img_saturation)    
    elif type=='h':
        img_hue = RandomHue(img)
        cv2.imshow('hue', img_hue)    

###
def randomBlur():
    img = cv2.imread('dog.jpg')
    cv2.imshow('original', img)
    #模糊处理，kernel=(5,5)越大模糊越严重
    img_blur = cv2.blur(img,(5,5))
    cv2.imshow('blur', img_blur)
    

if __name__ == '__main__':
    '''
    for i in range(10):
        test_hsv('s')
        cv2.waitKey(1000)
    '''
    randomBlur()

    while cv2.waitKey(1) != ord('q'):
        pass
    exit(0)
