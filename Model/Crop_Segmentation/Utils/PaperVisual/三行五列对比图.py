import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


gt_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm/predict_Unet3/gt'
pred_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm/predict_Unet3/pred'
image_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm/augmented'

gt = []
pred = []
image = []

# 设置图像显示的行数和列数
rows = 3
cols = 3

for root, dirs, files in os.walk(gt_path):
    random_numbers = random.sample(range(len(files)), cols)
    print(random_numbers)
    random_numbers = [826, 521, 446]
    # 826， 521，446
    for i in random_numbers:
        print(files[i])
        gt.append(os.path.join(gt_path, files[i]))
        pred.append(os.path.join(pred_path, files[i]))
        image.append(os.path.join(image_path, files[i]))

# 创建一个新的图像窗口
fig, axs = plt.subplots(rows, cols, figsize=(10, 6))

# 遍历每一行
for i in range(rows):
    # 遍历每一列
    for j in range(cols):
        # 计算当前图像在列表中的索引

        # 读取原图像
        image_path = image[j]
        img = mpimg.imread(image_path)

        # 读取label图像
        gt_path = gt[j]
        gt_img = mpimg.imread(gt_path)

        # 读取predict图像
        pred_path = pred[j]
        pred_img = mpimg.imread(pred_path)

        # 在每一行最左边的子图上添加标注文本
        if j == 0:
            if i == 0:
                axs[i, j].text(-0.2, 0.5, 'Original', transform=axs[i, j].transAxes,
                               fontsize=12, ha='center', va='center', rotation=90)
            elif i == 1:
                axs[i, j].text(-0.2, 0.5, 'Label', transform=axs[i, j].transAxes,
                               fontsize=12, ha='center', va='center', rotation=90)
            elif i == 2:
                axs[i, j].text(-0.2, 0.5, 'Prediction', transform=axs[i, j].transAxes,
                               fontsize=12, ha='center', va='center', rotation=90)

        if i == 0:
            # 绘制原图
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
        elif i == 1:
            # 绘制label图像
            axs[i, j].imshow(gt_img, cmap='gray')
            axs[i, j].axis('off')
        else:
            # 绘制predict图像
            axs[i, j].imshow(pred_img, cmap='gray')
            axs[i, j].axis('off')

    # 调整子图之间的间距，让图片尽量贴紧

# 调整子图之间的间距，让图片尽量贴紧
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# 显示图像
plt.show()

