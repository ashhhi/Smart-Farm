import numpy as np
from PIL import Image
import cv2


def adaptiveDirectionFilter(kernal_size=7, strides=1, angle_interval=30, threshold=0.8):
    """
    :param kernal_size:
    :param angle_interval:
    :return:
    """
    assert not 180 % angle_interval, 'angle_interval should be able to split 180 degrees equally.'
    assert kernal_size % 2, 'kernal_size should be odd.'
    assert kernal_size >= strides, 'kernal_size should larger than or equal to strides.'

    # calculate filter kernal
    mask = np.zeros((kernal_size, kernal_size), dtype=np.uint8)
    mask[:, kernal_size//2] = 1
    k = np.expand_dims(mask.copy(), axis=0)
    for angle in range(angle_interval, 180, angle_interval):
        matrix = cv2.getRotationMatrix2D((kernal_size//2, kernal_size//2), angle, 1)
        rotated_mask = cv2.warpAffine(mask, matrix, mask.shape[::-1])
        rotated_mask = np.expand_dims(rotated_mask, axis=0)
        k = np.append(k, rotated_mask, axis=0)
        # 显示图像
        # pil_image.show()


    return k

if __name__ == '__main__':
    k = adaptiveDirectionFilter(55, 1, 30, threshold=0.8)
    k *= 255
    for i in range(len(k)):
        img = k[i]
        cv2.imwrite(f'{i}_direction.jpg', img)
    # res_image = Image.fromarray(result)
    # res_image.show()
