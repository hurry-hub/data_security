#!/usr/bin/env python
# coding=utf-8

"""
@author: Richard Huang
@license: WHU
@contact: 2539444133@qq.com
@file: hog_classify.py
@date: 22/03/31 16:38
@desc: 
"""

import os
import cv2
import numpy as np
import glob
from imutils.object_detection import non_max_suppression


def get_svm_detector(SVM):
    """
    获得SVM分类器的支持向量和决策函数系数
    :param SVM: SVM解释器
    :return: 支持向量和决策函数系数
    """
    sv = SVM.getSupportVectors()            # SVM支持向量
    rho, _, _ = SVM.getDecisionFunction(0)  # SVM的决策函数
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)       # 得到支持向量和决策函数系数的一个数组

def read_neg_samples(foldername):
    """
    读取负样本信息
    :param foldername:  负样本文件夹路径
    :return: 负样本图片信息及标签
    """
    imgs = []
    labels = []
    neg_count = 0
    filenames = glob.iglob(os.path.join(foldername, '*'))
    for filename in filenames:
        src = cv2.imread(filename)
        src = cv2.resize(src, (96, 160))
        imgs.append(src)
        labels.append(-1)
        neg_count += 1

    return imgs, labels

def read_pos_samples(foldername):
    """
    读取正样本信息
    :param foldername:  正样本文件夹路径
    :return:  正样本图片信息及标签
    """
    imgs = []
    labels = []
    pos_count = 0
    filenames = glob.iglob(os.path.join(foldername, '*'))

    for filename in filenames:
        src = cv2.imread(filename)
        src = cv2.resize(src, (96, 160))
        imgs.append(src)
        labels.append(1)
        pos_count += 1

    return imgs, labels

def get_features(features, labels):
    """
    获得训练集信息
    :param features: 待填充的训练集图片信息
    :param labels:  待填充的训练集图片标签
    :return:  读取完成的训练集图片信息及标签
    """
    pos_imgs, pos_labels = read_pos_samples('updated_ninria/train/pos')
    computeHog(pos_imgs, features)
    [labels.append(1) for _ in range(len(pos_imgs))]

    neg_imgs, neg_labels = read_neg_samples('updated_ninria/train/neg')
    computeHog(neg_imgs, features)
    [labels.append(-1) for _ in range(len(neg_imgs))]

    return features, labels

def computeHog(imgs, features, wsize=(128, 64)):
    """
    计算HOG特征
    :param imgs: 待计算图片信息
    :param features: 将要得出的HOG特征
    :param wsize: 窗口大小
    :return:
    """
    hog = cv2.HOGDescriptor()
    count = 0

    for i in range(len(imgs)):
        if imgs[i].shape[1] >= wsize[1] and imgs[i].shape[0] >= wsize[0]:
            y = imgs[i].shape[0] - wsize[0]
            x = imgs[i].shape[1] - wsize[1]
            h = imgs[i].shape[0]
            w = imgs[i].shape[1]
            roi = imgs[i][y: y + h, x: x + w]
            features.append(hog.compute(roi))
            count += 1

    print('count = ', count)

def test(Hog, filepath):
    """
    :param Hog: hog及SVM分类器
    :param filepath: 测试文件夹路径
    :return: none
    """
    for file in os.listdir(filepath):
        imageSrc = cv2.imread(filepath + file)
        imageSrc = cv2.resize(imageSrc, (96, 160))
        # 输入检测的图像
        rects, scores = Hog.detectMultiScale(imageSrc, 0, winStride=(8, 8), padding=(0, 0), scale=1.05)
        sc = [score[0] for score in scores]
        sc = np.array(sc)

        # 转换下输出格式(x,y,w,h) -> (x1,y1,x2,y2)
        for i in range(len(rects)):
            r = rects[i]
            rects[i][2] = r[0] + r[2]
            rects[i][3] = r[1] + r[3]

        pick = []
        # 非极大值移植
        print('rects_len', len(rects))
        pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)
        print('pick_len = ', len(pick))

        # 画出矩形框
        for (x, y, w, h) in pick:
            cv2.rectangle(imageSrc, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 对检测出来的分类为1的矩形进行遍历框出来
        cv2.imshow('dst', imageSrc)
        cv2.waitKey(0)



if __name__ == "__main__":
    svm = cv2.ml.SVM_create()
    # SVM的参数初始化，这里用SVM线性分类器来做速度比较快
    # SVM对象参数的设置，核算子的设定等
    svm.setCoef0(0)
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)
    svm.setC(0.01)
    svm.setType(cv2.ml.SVM_EPS_SVR)

    hog = cv2.HOGDescriptor()
    train_data = []
    train_labels = []
    get_features(train_data, train_labels)
    # 把计算得到的HOG数据放到SVM对象分类器里面进行训练
    svm.train(np.array(train_data), cv2.ml.ROW_SAMPLE, np.array(train_labels))
    hog.setSVMDetector(get_svm_detector(svm))
    test(hog, "updated_ninria\\train\\neg\\")
    # test(hog, "updated_ninria\\test\\pos\\")