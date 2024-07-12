import cv2
import numpy as np

stem_mask_path = '/Users/shijunshen/Documents/Code/PycharmProjects/Smart-Farm/Model/Crop_Segmentation/Morphology/3691716968414_-pic_jpg.rf.b8f9d2f1e596dfce7d2b80ec8dc044a5.png'
original_image = '/Users/shijunshen/Documents/Code/PycharmProjects/Smart-Farm/Model/Crop_Segmentation/Morphology/3691716968414_-pic_jpg.rf.b8f9d2f1e596dfce7d2b80ec8dc044a5.jpg'


# 读取图像
image = cv2.imread(original_image, 0)

# 定义结构元素的大小（用于形态学操作）
kernel_size = (11, 11)

# 进行高帽滤波
tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size))

# 显示原始图像和高帽滤波结果

image_copy = image.copy()
darkened_pixels = np.where(tophat > 40)
image_copy[darkened_pixels] = image_copy[darkened_pixels] * 0  # 调整乘数以控

# 显示结果图像
cv2.imshow('Original Image', image)
cv2.imshow('Top-Hat Filter Result', tophat)
cv2.imshow('Result', image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
