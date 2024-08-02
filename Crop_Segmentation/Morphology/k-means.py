stem_mask_path = '/Users/shijunshen/Documents/Code/PycharmProjects/Smart-Farm/Model/Crop_Segmentation/Morphology/3691716968414_-pic_jpg.rf.b8f9d2f1e596dfce7d2b80ec8dc044a5.png'
original_image = '/Users/shijunshen/Documents/Code/PycharmProjects/Smart-Farm/Model/Crop_Segmentation/Morphology/3691716968414_-pic_jpg.rf.b8f9d2f1e596dfce7d2b80ec8dc044a5.jpg'

import cv2
import numpy as np
from sklearn.cluster import KMeans

# 读取图像
image = cv2.imread(original_image)

# 将图像转换为RGB颜色空间
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像转换为二维数组
pixel_values = image.reshape((-1, 3))

# 定义K-means算法的聚类数
k = 3

# 创建K-means模型并进行训练
kmeans = KMeans(n_clusters=k)
kmeans.fit(pixel_values)

# 获取每个像素所属的聚类标签
labels = kmeans.labels_

# 获取每个聚类的中心颜色值
colors = kmeans.cluster_centers_

# 创建一个具有相同形状的数组，用于存储每个像素的颜色
new_image = np.zeros_like(pixel_values)

# 根据聚类标签，将每个像素设置为对应的中心颜色
for i in range(len(pixel_values)):
    new_image[i] = colors[labels[i]]

# 将图像恢复为原始形状
new_image = new_image.reshape(image.shape)

# 将图像转换为uint8数据类型
new_image = np.uint8(new_image)

# 显示聚类结果图像
cv2.imshow('Clustered Image', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
