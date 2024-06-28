import numpy as np
from tqdm import tqdm

image_path = ''
label_path = r'/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Smart-Farm/augmented_mask'

Class_Map = {
    'Background': [0, 0, 0],
    'Leaf': [0, 255, 0],
    'Stem': [0, 0, 255],
}


import os
def Dataloader(image_path, label_path):
    g = os.walk(label_path)
    image = []
    label = []
    image_name = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if file_name.split('.')[-1] == 'png':
                label.append(str(os.path.join(path, file_name)))
                image_name.append(file_name)

    # 处理xml文件
    return image, label, image_name

_, label_path, _ = Dataloader(image_path, label_path)

# label_path = ['/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Ash-Roboflow/Version1/mask/3651716968391_-pic_jpg.rf.5e8b8d70005352f36899c6344ef1c924.png']
from PIL import Image
import cv2 as cv
import numpy as np

temp = np.array(list(reversed(Class_Map['Background'])))
print(temp)

img = cv.imread(label_path[0])
background_mask = np.full_like(img, np.array(list(reversed(Class_Map['Background']))))
leaf_mask = np.full_like(img, np.array(list(reversed(Class_Map['Leaf']))))
stem_mask = np.full_like(img, np.array(list(reversed(Class_Map['Stem']))))

background_pixel_sum = 0
leaf_pixel_sum = 0
stem_pixel_sum = 0


for file in tqdm(label_path):
    img = cv.imread(file)
    background_pixel_sum += np.sum(np.all(background_mask == img, axis=-1))
    leaf_pixel_sum += np.sum(np.all(leaf_mask == img, axis=-1))
    stem_pixel_sum += np.sum(np.all(stem_mask == img, axis=-1))
    # print(background_pixel + stem_pixel + leaf_pixel)
    print('background:', background_pixel_sum)
    print('leaf', leaf_pixel_sum)
    print('stem', stem_pixel_sum)
    print()