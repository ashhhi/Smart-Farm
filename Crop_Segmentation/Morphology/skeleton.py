import cv2
import numpy as np

# 读取骨架化图像
image = cv2.imread('/Users/shijunshen/Documents/Code/PycharmProjects/Smart-Farm/Model/Crop_Segmentation/Morphology/3691716968414_-pic_jpg.rf.b8f9d2f1e596dfce7d2b80ec8dc044a5.png', 0)

# 进行骨架化处理
skeleton = cv2.ximgproc.thinning(image)

# 进行距离变换
distance_transform = cv2.distanceTransform(image, cv2.DIST_L2, 3)

# 选择阈值来确定树干区域
distance_transform[skeleton == 0] = 0

# 计算树干的平均粗度
mean_thickness = np.mean(distance_transform[distance_transform > 0]) * 2

# 输出树干的平均粗度
print('Mean Thickness:', mean_thickness)

# 显示结果
cv2.imwrite('skel1.jpg', image)
cv2.imwrite('skel2.jpg', skeleton)
# cv2.imshow('Binary Image', image)
# cv2.imshow('Skeleton Image', skeleton)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

