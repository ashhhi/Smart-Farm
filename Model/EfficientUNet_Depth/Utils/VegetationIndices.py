import cv2
import numpy as np
import matplotlib.pyplot as plt


class VegetationIndices:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        img1 = np.array(self.image, dtype='int')  # 转换成int型，不然会导致数据溢出
        self.B, self.G, self.R = cv2.split(img1)
        Bn = self.B / 255
        Gn = self.G / 255
        Rn = self.R / 255
        self.b = Bn / (Bn + Gn + Rn)
        self.g = Gn / (Bn + Gn + Rn)
        self.r = Rn / (Bn + Gn + Rn)

    def show(self, res):
        res = np.array(res, dtype='uint8')  # 重新转换成uint8类型
        ret2, th2 = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        plt.figure(figsize=(10, 5), dpi=80)
        plt.subplot(131), plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)), \
        plt.title('Original'), plt.axis('off')
        plt.subplot(132), plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB)), \
        plt.title('ExG_gray'), plt.axis('off')
        plt.subplot(133), plt.imshow(cv2.cvtColor(th2, cv2.COLOR_BGR2RGB)), \
        plt.title('OTSU_bw'), plt.axis('off')
        plt.show()

    def ExG(self, isShow=True):
        res = 2 * self.g - self.r - self.b
        [m, n] = res.shape

        for i in range(m):
            for j in range(n):
                if res[i, j] < 0:
                    res[i, j] = 0
                elif res[i, j] > 1:
                    res[i, j] = 1

        res *= 255
        if isShow:
            self.show(res)

        return res

    def ExGR(self, isShow=True):
        ExRed = 1.4 * self.r - self.g
        res = self.ExG(isShow=False) - ExRed * 255
        if isShow:
            self.show(res)
        return res

    def NDI(self, isShow=True):
        res = 128 * (((self.G - self.R) / (self.G + self.R)) + 1)
        if isShow:
            self.show(res)
        return res

test = VegetationIndices('/Users/shijunshen/Documents/dataset/02 Testing Pool/Image_and_labels/Dataset/train/bokchoy_day_6_2_png.rf.8b54c9b8520710773901e4c0476df772.jpg')
test.NDI()