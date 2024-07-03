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

with open('/Users/shijunshen/Documents/Code/PycharmProjects/Smart-Farm/Model/Crop_Segmentation/config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Train']['Image']['Width']
    Height = yaml_data['Train']['Image']['Height']

image_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Smart-Farm/image/0629_png_jpg.rf.35b395813db393b96f6679ca8f28cd15.jpg'
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
    distances = np.abs(cv2.pointPolygonTest(contour, (x, y), True))
    thickness = np.mean(distances)
    thicknesses.append(thickness)

# Average thickness of stems
average_thickness = np.mean(thicknesses)
print(average_thickness)


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