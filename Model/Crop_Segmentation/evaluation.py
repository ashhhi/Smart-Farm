
import os

import cv2
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import os
import torchvision.transforms as transforms

def sorensen_dices(y_true, y_pred):
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    intersection = np.sum(y_true * y_pred)
    TN = np.sum((1 - y_true) * (1 - y_pred))
    TP = intersection
    FN = np.sum(y_true * (1 - y_pred))
    FP = np.sum((1 - y_true) * y_pred)
    FPR = FP / (TN + FP)
    if np.isnan(FPR):
        FPR = np.int64(0)
    FNR = FN / (TP + FN)
    if np.isnan(FNR):
        FNR = np.int64(0)
    IOU = TP / (TP + FP + FN)
    if np.isnan(IOU):
        IOU = np.int64(0)
    precision = TP / (TP + FP)
    if np.isnan(precision):
        precision = np.int64(0)
    specificity = TN / (TN + FP)
    if np.isnan(specificity):
        specificity = np.int64(0)
    recall = TP / (TP + FN)
    if np.isnan(recall):
        recall = np.int64(0)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    if np.isnan(accuracy):
        accuracy = np.int64(0)
    f1 = (2 * precision * recall) / (precision + recall)
    if np.isnan(f1):
        f1 = np.int64(0)

    return FPR, FNR, precision, specificity, recall, accuracy, f1, IOU


# 进行评估，获得评估结果
def Evaluation(prediction, ground_truth):  # 输出三个评测指标（整体上）
    # 获取文件夹下的文件列表
    length = prediction.shape[0]
    class_num = np.max(ground_truth) + 1
    print(length, class_num)
    # 遍历文件列表
    FPRs = 0
    FNRs = 0
    precisions=0
    specificitys=0
    recalls=0
    accuarys=0
    f1s = 0
    ious = 0
    # 对每一个类循环
    for c in range(class_num):
        # 对每一张图片循环
        for i in tqdm(range(len(prediction))):
            pred_arr = prediction[i].copy()
            gt_arr = ground_truth[i].copy()
            pred_arr[prediction[i] != c] = 0
            pred_arr[prediction[i] == c] = 1
            gt_arr[ground_truth[i] != c] = 0
            gt_arr[ground_truth[i] == c] = 1

            FPR, FNR, precision, specificity, recall, accuracy, f1, IOU = sorensen_dices(gt_arr, pred_arr)

            precisions += precision
            f1s += f1
            FPRs = FPRs + FPR
            FNRs = FNRs + FNR
            specificitys = specificitys + specificity
            recalls = recalls + recall
            accuarys = accuarys + accuracy
            ious = ious + IOU


    total = length * class_num

    FPRsm = FPRs / total
    FNRsm = FNRs / total
    precisionsm = precisions / total
    specificitysm = specificitys / total
    recallsm = recalls / total
    accuarysm = accuarys / total
    f1sm = f1s / total
    iousm = ious / total
    output = "FPR: %f,\nFNR: %f,\nrecall: %f,\nprecision: %f,\nPA: %f,\nf1: %f,\nmiou: %f\nspecificity:%f" % (FPRsm, FNRsm, recallsm, precisionsm, accuarysm, f1sm, iousm, specificitysm)
    print(output)
    return

# test()
# Evaluation(Predict_Path=r"/Users/shijunshen/Desktop/pred/againnnnnnn/UNet/pred", Label_Path='/Users/shijunshen/Desktop/pred/againnnnnnn/UNet/gt')