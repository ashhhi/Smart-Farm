import numpy as np
from PIL import Image

gt_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm/predict/gt'
pred_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm/predict/pred'
num_classes = 3

def compute():
    iou_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for i in range(2256):
        # 读取gt和pred图像
        gt_image = Image.open(gt_path + f"/{i}.png")  # 替换为gt图像的路径和命名规则
        pred_image = Image.open(pred_path + f"/{i}.png")  # 替换为pred图像的路径和命名规则

        # 将图像转换为numpy数组
        gt_array = np.ravel(np.argmax(np.array(gt_image), axis=-1))
        pred_array = np.ravel(np.argmax(np.array(pred_image), axis=-1))

        # 计算每个类别的TP、FP和FN
        iou_class = []
        precision_class = []
        recall_class = []
        f1_class = []
        for cls in range(num_classes):
            tp = np.sum((gt_array == cls) & (pred_array == cls))
            fp = np.sum((gt_array != cls) & (pred_array == cls))
            fn = np.sum((gt_array == cls) & (pred_array != cls))
            tn = np.sum((gt_array != cls) & (pred_array != cls))

            # IoU
            tmp = tp+fp+fn
            if tmp == 0:
                continue
            else:
                iou = tp / tmp
                iou_class.append(iou)

            # Precision
            tmp = tp + fp
            if tmp == 0:
                continue
            else:
                precision = tp / tmp
                precision_class.append(precision)

            # Recall
            tmp = tp + fn
            if tp+fn == 0:
                continue
            else:
                recall = tp / (tmp)
                recall_class.append(recall)

            # f1
            tmp = precision + recall
            if tmp == 0:
                continue
            else:
                f1 = 2 * (precision * recall) / tmp
                f1_class.append(f1)


        # 类别平均IoU
        mean_iou = np.mean(iou_class)
        iou_list.append(mean_iou)
        # 类别平均Precision
        mean_precision = np.mean(precision_class)
        precision_list.append(mean_precision)
        # 类别平均Recall
        mean_recall = np.mean(recall_class)
        recall_list.append(mean_recall)
        # F1
        mean_f1 = np.mean(f1_class)
        f1_list.append(mean_f1)


    # 所有图像的平均IoU
    average_iou = np.mean(iou_list)
    average_precision = np.mean(precision_list)
    average_recall = np.mean(recall_list)
    average_f1 = np.mean(f1_list)
    return average_iou, average_precision, average_recall, average_f1




print(compute())