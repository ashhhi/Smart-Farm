import numpy as np
from PIL import Image
import cv2


def adaptiveDirectionFilter(image, kernal_size=7, strides=1, angle_interval=30, threshold=0.8):
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

    # Slip along the image
    padding_y = int((kernal_size + (image.shape[0]-1) * strides - image.shape[0]) / 2)
    padding_x = int((kernal_size + (image.shape[1]-1) * strides - image.shape[1]) / 2)

    padded_image = cv2.copyMakeBorder(image, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT, value=0)
    res_dilate = np.zeros_like(image)
    index_i = 0
    index_j = 0

    eros_index = []

    for i in range(kernal_size//2, padded_image.shape[0]-kernal_size//2, strides):
        index_j = 0
        for j in range(kernal_size//2, padded_image.shape[1]-kernal_size//2, strides):
            left_top_corner = [i-kernal_size//2, j-kernal_size//2]
            right_bottom_corner = [i+kernal_size//2+1, j+kernal_size//2+1]
            field = padded_image[left_top_corner[0]:right_bottom_corner[0], left_top_corner[1]:right_bottom_corner[1]]
            field = np.expand_dims(field, axis=0)
            tmp = np.sum((field * k).reshape(-1, k.shape[0]), axis=0)
            max_direction_index = np.argmax(tmp)
            max_direction_value = tmp[max_direction_index] / np.count_nonzero(k[max_direction_index])
            vertical_max_direction_index = int((np.argmax(tmp) + (k.shape[-1]+1) / 2) % (k.shape[0]))
            vertical_max_direction_value = tmp[vertical_max_direction_index] / np.count_nonzero(k[vertical_max_direction_index])

            max_direction_k = k[max_direction_index]
            if threshold * max_direction_value > vertical_max_direction_value:
                Sl = max_direction_k[kernal_size // 2 - kernal_size // 4:kernal_size // 2 + kernal_size // 4 + 1, kernal_size // 2 - kernal_size // 4:kernal_size // 2 + kernal_size // 4 + 1]
                f = padded_image[i-kernal_size // 4: i+kernal_size // 4 + 1, j-kernal_size // 4: j+kernal_size // 4 + 1]
                # 膨胀
                tmp = np.max(Sl*f)
                res_dilate[index_i][index_j] = tmp
                eros_index.append([i, j, index_i, index_j, max_direction_index])
            else:
                res_dilate[index_i][index_j] = image[index_i][index_j]
                pass
            index_j += 1
        index_i += 1
        print(i, j)

    res_eros = res_dilate.copy()
    res_dilate_padding = cv2.copyMakeBorder(res_dilate, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT, value=0)
    # 腐蚀
    for item in eros_index:
        i, j, index_i, index_j, max_direction_index = item
        max_direction_k = k[max_direction_index]
        Sl = max_direction_k[kernal_size // 2 - kernal_size // 4:kernal_size // 2 + kernal_size // 4 + 1, kernal_size // 2 - kernal_size // 4:kernal_size // 2 + kernal_size // 4 + 1]
        f = res_dilate_padding[i - kernal_size // 4: i + kernal_size // 4 + 1, j - kernal_size // 4: j + kernal_size // 4 + 1]
        tmp = np.min(Sl * f)
        res_eros[index_i][index_j] = tmp
    print(res_eros.shape)
    return res_eros

if __name__ == '__main__':
    origin_image = cv2.imread('/Users/shijunshen/Documents/Code/PycharmProjects/Smart-Farm/Model/Crop_Segmentation/Morphology/3691716968414_-pic_jpg.rf.b8f9d2f1e596dfce7d2b80ec8dc044a5.png', 0)
    origin_image = cv2.resize(origin_image, (320, 320))

    result = adaptiveDirectionFilter(origin_image, 55, 1, 5, threshold=0.8)
    res_image = Image.fromarray(result)
    res_image.show()

    origin_image = Image.fromarray(origin_image)
    origin_image.show()
