import cv2
import numpy as np

stem_mask_path = '/Users/shijunshen/Documents/Code/PycharmProjects/Smart-Farm/Model/Crop_Segmentation/Morphology/3691716968414_-pic_jpg.rf.b8f9d2f1e596dfce7d2b80ec8dc044a5.png'
original_image = '/Users/shijunshen/Documents/Code/PycharmProjects/Smart-Farm/Model/Crop_Segmentation/Morphology/3691716968414_-pic_jpg.rf.b8f9d2f1e596dfce7d2b80ec8dc044a5.jpg'


# 读取图像
image = cv2.imread(original_image, 0)

# 定义结构元素的大小（用于形态学操作）
kernel_size = (5, 5)

# 进行黑帽滤波
blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size))

# 显示原始图像和黑帽滤波结果
cv2.imshow('Original Image', image)
cv2.imshow('Black-Hat Filter Result', blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()