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

image_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/XIAOMICamera/6_29/PIC_20240628_102600080.jpg'
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


for [vx, vy, x, y] in line_parameter:
    # Calculate two points on the line for drawing
    point1 = (x - 1000 * vx, y - 1000 * vy)
    point2 = (x + 1000 * vx, y + 1000 * vy)

    point1 = (int(point1[0]), int(point1[1]))
    point2 = (int(point2[0]), int(point2[1]))
    # Draw the line on the image
    cv2.line(stem_region, point1, point2, (127), 2)

cv2.imshow('123', stem_region)
cv2.waitKey()
#
# model = tf.keras.models.load_model('removeBackground.h5', custom_objects={"mIoU": mIoU})
# probability_vector = model.predict(image)
# max_prob_class_pot = np.argmax(probability_vector, axis=-1)
#
# max_prob_class_pot = max_prob_class_pot.squeeze()
# max_prob_class_pot[max_prob_class_pot==1] = 255
# max_prob_class_pot = np.array(max_prob_class_pot, dtype=np.uint8)
#
# cv2.imshow('123', max_prob_class_pot)
# cv2.waitKey()