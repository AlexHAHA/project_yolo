import xml.etree.ElementTree as ET
import os

VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        difficult = int(obj.find('difficult').text)
        if difficult == 1:
            # print(filename)
            continue
        obj_struct['name'] = obj.find('name').text
        #obj_struct['pose'] = obj.find('pose').text
        #obj_struct['truncated'] = int(obj.find('truncated').text)
        #obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)

    return objects

''' 训练集voc2007 '''
txt_file = open('voc2007train.txt', 'w')  # 存放训练集的image及label
train_file = open(r'D:\CETCA_DeepLearning\datasets\voc\VOCdevkit\VOC2007\ImageSets\Main\train.txt', 'r') # 获取训练集包含的image
lines = train_file.readlines()

''' 测试集voc2007 '''
#txt_file = open('voc2007test.txt', 'w')  # 存放测试集的image及label
#test_file = open('voc07testimg.txt', 'r') # 获取测试集包含的image
#lines = test_file.readlines()


# 数据集包含的image
lines = [x[:-1] for x in lines]

path_base = r'D:\CETCA_DeepLearning\datasets\voc\VOCdevkit\VOC2007'
path_annotations = os.path.join(path_base, 'Annotations')
path_images    = os.path.join(path_base, 'JPEGImages')
xml_files = os.listdir(path_annotations)
count = 0
for xml_file in xml_files:
    count += 1
    # 若xml file不在数据集中
    if xml_file.split('.')[0] not in lines:
        #print(f'Warning:dataset not contain {xml_file.split('.')[0]}')
        continue
    image_name = xml_file.split('.')[0] + '.jpg'
    image_path = os.path.join(path_images, image_name)
    results = parse_rec(os.path.join(path_annotations, xml_file))
    if len(results) == 0:
        print(f'Warning: {xml_file} has no right objects')
        continue
        
    #txt_file.write(image_path)
    txt_file.write(image_name)
    # num_obj = len(results)
    for result in results:
        class_name = result['name']
        bbox     = result['bbox']
        class_id  = VOC_CLASSES.index(class_name)
        txt_file.write(' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + ' ' + str(class_id))
        
    txt_file.write('\n')
    
txt_file.close()





























