import cv2
import tensorflow as tf
import yaml
import numpy as np
from removeBackground import removebackground
from PIL import Image
def mIoU(y_true, y_pred):
    true = tf.reshape(tf.argmax(y_true, axis=-1), [-1])
    pred = tf.reshape(tf.argmax(y_pred, axis=-1), [-1])
    cm = tf.math.confusion_matrix(true, pred, dtype=tf.float32)
    diag_item = tf.linalg.diag_part(cm)
    # return IoU_per
    mIoU = tf.reduce_mean(diag_item / (tf.reduce_sum(cm, 0) + tf.reduce_sum(cm, 1) - tf.linalg.diag_part(cm)))

    # tf.summary.scalar('mean IoU', mIoU)
    return mIoU

def calculate_average_distance(points, line_params):
    """
    计算点集中所有点到直线的最短距离的平均值

    参数：
    points：点集，形状为 (N, 2)，每个点的坐标形式为 [x, y]
    line_params：拟合直线的参数，形式为 [vx, vy, x, y]

    返回值：
    average_distance：平均距离
    """

    vx, vy, x, y = line_params

    # 计算直线的法向量
    line_normal = np.array([-vy, vx])

    distances = []
    for point in points:
        point_vector = np.array([point[0][0] - x, point[0][1] - y])
        # 计算点到直线的距离（点到直线的投影长度）
        distance = np.abs(np.dot(point_vector, line_normal))
        distances.append(distance)

    average_distance = np.mean(distances)

    return average_distance

with open('/Users/shijunshen/Documents/Code/PycharmProjects/Smart-Farm/Model/Crop_Segmentation/config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Train']['Image']['Width']
    Height = yaml_data['Train']['Image']['Height']

image_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Ice Plant(different resolution)/1600*1200/3671716968397_.pic.jpg'
image_origin = tf.keras.preprocessing.image.load_img(image_path, target_size=(Height, Width))
image_origin = tf.keras.preprocessing.image.img_to_array(image_origin)
image_origin = np.expand_dims(image_origin, axis=0)
image = image_origin / 255.0  # 标准化图像像素值



model = tf.keras.models.load_model('EfficientUnet3Plus.h5', custom_objects={"mIoU": mIoU})

probability_vector = model.predict(image)
# print(probability_vector.shape)

max_prob_class_stem = np.argmax(probability_vector, axis=-1)



# Extract stem region
stem_region = np.where(max_prob_class_stem == 2, 255, 0).astype(np.uint8).squeeze()

# 进行骨架化处理
skeleton = cv2.ximgproc.thinning(stem_region)

# 进行距离变换
distance_transform = cv2.distanceTransform(stem_region, cv2.DIST_L2, 3)

# 选择阈值来确定树干区域
distance_transform[skeleton == 0] = 0

# 计算树干的平均粗度
mean_thickness = np.mean(distance_transform[distance_transform > 0]) * 2

# 输出树干的平均粗度
print('Mean Thickness:', mean_thickness)

# model = tf.keras.models.load_model('RB_Iceplant.h5', custom_objects={"mIoU": mIoU})
# probability_vector = model.predict(image)
# max_prob_class_pot = np.argmax(probability_vector, axis=-1)
# max_prob_class_pot = max_prob_class_pot.squeeze()

max_prob_class_pot = removebackground(image).squeeze()

max_prob_class_pot[max_prob_class_pot == 1] = 255
max_prob_class_pot = np.array(max_prob_class_pot, dtype=np.uint8)

'''
先腐蚀后膨胀，既保持原有大小，又分离两个植物盆
'''
# 定义腐蚀操作的内核大小和形状
kernel = np.ones((5, 5), np.uint8)

# 进行开操作
iterations = 1  # 根据需要调整迭代次数
# max_prob_class_pot = cv2.erode(max_prob_class_pot, kernel, iterations=iterations)
# max_prob_class_pot = cv2.dilate(max_prob_class_pot, kernel, iterations=iterations)
max_prob_class_pot = cv2.morphologyEx(max_prob_class_pot,cv2.MORPH_CLOSE, kernel, iterations=iterations)


'''
方框检测，框出目标的盆子
'''
contours, _ = cv2.findContours(max_prob_class_pot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=lambda arr: len(arr))

x, y, width, height = cv2.boundingRect(largest_contour)

print(x, y, width, height)
cv2.rectangle(max_prob_class_pot, (x, y), (x + width, y + height), 127)

'''
测量出实际的盆子长宽为：50cm，25cm，也就是2:1
'''
from sympy import symbols, Eq, solve, sin, cos
# 解方程
x = symbols('x')        # x 是盆子的宽，那么 2x 是盆子的长
alpha = symbols('alpha')
equation1 = Eq(cos(alpha) * x + 2 * x * sin(alpha), height)
equation2 = Eq(sin(alpha) * x + 2 * x * cos(alpha), width)
solution = solve((equation1, equation2), (x, alpha))
x = solution[0][0].evalf()

ratio = x / 25      # 25 是以cm为单位的，因此ratio的单位是 像素/厘米
stem_thickness_real = mean_thickness / ratio
print('stem thickness(cm): ', stem_thickness_real)

cv2.imwrite('image_save.jpg', max_prob_class_pot)
cv2.imshow('pot', max_prob_class_pot)
img_origin = cv2.imread(image_path)
cv2.imshow('origin', img_origin)
cv2.waitKey()