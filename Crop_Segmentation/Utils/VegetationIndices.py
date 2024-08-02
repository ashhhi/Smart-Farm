import cv2
import numpy as np
import matplotlib.pyplot as plt


class VegetationIndices:
    def __init__(self, image):
        self.image = image
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
        plt.title('Processed_Gray'), plt.axis('off')
        plt.subplot(133), plt.imshow(cv2.cvtColor(th2, cv2.COLOR_BGR2RGB)), \
        plt.title('OTSU_bw'), plt.axis('off')
        plt.show()

    def ExG(self, isShow=False):
        res = 2 * self.g - self.r - self.b

        if isShow:
            res = (res + 1) * 127.5
            self.show(res)

        return res

    def ExGR(self, isShow=False):
        ExRed = 1.4 * self.r - self.g
        res = self.ExG() - ExRed

        if isShow:
            res = (res + 1) * 127.5
            self.show(res)

        return res

if __name__ == '__main__':


    image_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Handle/Ice Plant(different resolution)/1600*1200/3721716968423_.pic.jpg'
    img = cv2.imread(image_path)

    test = VegetationIndices(img)
    # test.NDI()

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ExG_img = (test.ExG() + 1) * 127.5
    ExGR_img = (test.ExGR() + 1) * 127.5
    # ExG_img = (test.ExG() + 1) / 2 * gray
    # ExG_img = np.array(ExG_img, dtype='uint8')
    # max_value = np.max(ExG_img)
    # ExG_img = ExG_img.astype(float) / max_value * 255

    # ExGR_img = (test.ExGR() + 1) / 2 * gray
    # ExGR_img = np.array(ExGR_img, dtype='uint8')
    # max_value = np.max(ExGR_img)
    # ExGR_img = ExG_img.astype(float) / max_value * 255

    ExG_img = np.array(ExG_img, dtype='uint8')  # 重新转换成uint8类型
    ExGR_img = np.array(ExGR_img, dtype='uint8')

    plt.figure(figsize=(10, 5), dpi=80)
    plt.subplot(141), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), \
    plt.title('Original'), plt.axis('off')
    plt.subplot(142), plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)), \
    plt.title('Gray'), plt.axis('off')
    plt.subplot(143), plt.imshow(cv2.cvtColor(ExG_img, cv2.COLOR_BGR2RGB)), \
    plt.title('Excess Green'), plt.axis('off')
    plt.subplot(144), plt.imshow(cv2.cvtColor(ExGR_img, cv2.COLOR_BGR2RGB)), \
    plt.title('Excess Green Red'), plt.axis('off')
    plt.show()
