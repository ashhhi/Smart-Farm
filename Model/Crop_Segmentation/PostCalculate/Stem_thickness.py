import cv2
import tensorflow as tf
import yaml
import numpy as np
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

image_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/XIAOMICamera/6_29/PIC_20240629_144416410.jpg'
image_origin = tf.keras.preprocessing.image.load_img(image_path, target_size=(Height, Width))
image_origin = tf.keras.preprocessing.image.img_to_array(image_origin)
image_origin = np.expand_dims(image_origin, axis=0)
image = image_origin / 255.0  # 标准化图像像素值



model = tf.keras.models.load_model('EfficientUnet3Plus7_200_epoch.h5')

probability_vector = model.predict(image)
# print(probability_vector.shape)

max_prob_class_stem = np.argmax(probability_vector, axis=-1)



# Extract stem region
stem_region = np.where(max_prob_class_stem == 2, 255, 0).astype(np.uint8).squeeze()

# Find contours
contours, _ = cv2.findContours(stem_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Compute thickness for each stem
thicknesses = []

line_parameter = []

for contour in contours:
    if len(contour) < 30:
        continue
    # Fit a line to the contour
    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    line_parameter.append([vx, vy, x, y])
    # Convert x and y to floating-point
    x = float(x)
    y = float(y)

    # Compute thickness

    # distances = np.abs(cv2.pointPolygonTest(contour, (x, y), True))
    # thickness = np.mean(distances)
    thickness = calculate_average_distance(points=contour, line_params=[vx, vy, x, y])
    thicknesses.append(thickness)

# Average thickness in pixels level
average_thickness_pixels = np.mean(thicknesses) * 2    # two sides
print('Thickness (pixels):', average_thickness_pixels)
#
#
# for [vx, vy, x, y] in line_parameter:
#     # Calculate two points on the line for drawing
#     point1 = (x - 1000 * vx, y - 1000 * vy)
#     point2 = (x + 1000 * vx, y + 1000 * vy)
#
#     point1 = (int(point1[0]), int(point1[1]))
#     point2 = (int(point2[0]), int(point2[1]))
#     # Draw the line on the image
#     cv2.line(stem_region, point1, point2, (127), 2)
#
# cv2.imshow('123', stem_region)
# cv2.waitKey()

model = tf.keras.models.load_model('NewNet.h5', custom_objects={"mIoU": mIoU})
probability_vector = model.predict(image)
max_prob_class_pot = np.argmax(probability_vector, axis=-1)

max_prob_class_pot = max_prob_class_pot.squeeze()
max_prob_class_pot[max_prob_class_pot==1] = 255
max_prob_class_pot = np.array(max_prob_class_pot, dtype=np.uint8)

'''
先腐蚀后膨胀，既保持原有大小，又分离两个植物盆
'''
# 定义腐蚀操作的内核大小和形状
kernel = np.ones((3, 3), np.uint8)

# 进行多次腐蚀操作
iterations = 10  # 根据需要调整腐蚀的次数
max_prob_class_pot = cv2.erode(max_prob_class_pot, kernel, iterations=iterations)
max_prob_class_pot = cv2.dilate(max_prob_class_pot, kernel, iterations=iterations)


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
stem_thickness_real = average_thickness_pixels / ratio
print('stem thickness(cm): ', stem_thickness_real)

cv2.imwrite('image_save.jpg', max_prob_class_pot)
cv2.imshow('123', max_prob_class_pot)
cv2.waitKey()