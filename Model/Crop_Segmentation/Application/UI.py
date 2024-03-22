import sys

import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import tensorflow as tf
import numpy as np
import yaml

with open('../config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Image']['Width']
    Height = yaml_data['Image']['Height']

# 加载模型（假设您已经有了一个可以进行图像预测的模型）
model = tf.keras.models.load_model('EfficientUnet_50.h5')


class ImagePredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Prediction App')
        self.image_label = QLabel()
        self.prediction_label = QLabel()  # 新增的 QLabel 控件
        self.upload_button = QPushButton('Upload Image')
        self.upload_button.clicked.connect(self.upload_image)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.prediction_label)  # 添加预测结果的 QLabel 控件
        layout.addWidget(self.upload_button)

        self.setLayout(layout)

    def upload_image(self):
        # 打开文件对话框以选择图像文件
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Select Image', '', 'Image Files (*.png *.jpg *.jpeg)')

        if file_path:
            # 加载选择的图像文件
            image_origin = tf.keras.preprocessing.image.load_img(file_path, target_size=(Height, Width))
            image_origin = tf.keras.preprocessing.image.img_to_array(image_origin)
            image_origin = np.expand_dims(image_origin, axis=0)
            image = image_origin / 255.0  # 标准化图像像素值

            # 进行图像预测
            probability_vector = model.predict(image)
            color_map = {
                0: [255, 0, 0],  # Class 0: background
                1: [0, 255, 0],  # Class 1: leaf
                2: [0, 0, 255]  # Class 2: stem
            }
            predicted_labels = np.argmax(probability_vector, axis=-1)
            colored_image = np.zeros((predicted_labels.shape[0], Height, Width, 3), dtype=np.uint8)
            for n in range(predicted_labels.shape[0]):
                for i in range(Height):
                    for j in range(Width):
                        label = predicted_labels[n, i, j]
                        colored_image[n, i, j] = color_map[label]

            # 转换预测结果为 QImage 对象
            # 显示图像和预测结果

            image = image_origin[0]
            image = image.astype(np.uint8)
            print(image.shape)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            image = QPixmap.fromImage(image)
            self.image_label.setPixmap(image.scaled(Width, Height))

            prediction = colored_image[0]
            height, width, channel = prediction.shape
            bytes_per_line = 3 * width
            prediction = QImage(prediction.data, width, height, bytes_per_line, QImage.Format_RGB888)
            prediction = QPixmap.fromImage(prediction)
            self.prediction_label.setPixmap(prediction.scaled(Width, Height))  # 在新的 QLabel 控件上显示预测结果








# 创建应用程序实例
app = QApplication(sys.argv)

# 创建界面实例并显示
window = ImagePredictionApp()
window.show()

# 运行应用程序事件循环
sys.exit(app.exec_())