#!/usr/bin/env python
# coding=utf-8

"""
@author: Richard Huang
@license: WHU
@contact: 2539444133@qq.com
@file: search.py
@date: 22/03/31 14:34
@desc: 
"""
import cv2
import os
import operator

import numpy as np

folder = "image\\Images\\"

if __name__ == "__main__":
    target = cv2.imread("image\\Images\\cannon\\029_0046.jpg")                              # 读取将要搜索的目标图片
    hist = cv2.calcHist([target], [0, 1, 2], None, [255, 255, 255], [0, 256, 0, 256, 0, 256])     # 计算出颜色直方图
    hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX).flatten()                     # 归一化，使范围在[0, 255]

    # 获取文件夹下所有文件名
    all_data = []
    files = os.walk(folder)
    for filepath, dirnames, filenames in files:
        for filename in filenames:
            all_data.append(os.path.join(filepath, filename))

    score_res_B = {}            # 巴氏距离分数
    score_res_C = {}            # 皮尔逊相关系数分数
    score_res_K = {}            # 卡方距离分数
    for i in range(len(all_data)):
        temp = cv2.imread(all_data[i])
        temp_hist = cv2.calcHist([temp], [0, 1, 2], None, [255, 255, 255], [0, 256, 0, 256, 0, 256])  # 读取文件
        temp_hist = cv2.normalize(temp_hist, temp_hist, 0, 255, cv2.NORM_MINMAX).flatten()      # 均衡化
        score1 = cv2.compareHist(hist, temp_hist, cv2.HISTCMP_BHATTACHARYYA)                    # 计算巴氏距离相关度
        score2 = cv2.compareHist(hist, temp_hist, cv2.HISTCMP_CORREL)                           # 计算皮尔逊相关系数相关度
        score3 = cv2.compareHist(hist, temp_hist, cv2.HISTCMP_CHISQR)                           # 计算卡方距离相关度
        score_res_B[all_data[i]] = score1
        score_res_C[all_data[i]] = score2
        score_res_K[all_data[i]] = score3

    # 排序
    sorted_score_B = sorted(score_res_B.items(), key=operator.itemgetter(1))
    sorted_score_C = sorted(score_res_C.items(), key=operator.itemgetter(1), reverse=True)
    sorted_score_K = sorted(score_res_K.items(), key=operator.itemgetter(1))
    print("巴氏距离计算得到的最相似图片为 {0}, 对应的相似度为 {1}, 越接近0越相似".format(sorted_score_B[0][0], sorted_score_B[0][1]))
    print("皮尔逊相关系数计算得到的最相似图片为 {0}, 对应的相似度为 {1}, 越接近1越相似".format(sorted_score_C[0][0], sorted_score_C[0][1]))
    print("卡方距离计算得到的最相似图片为 {0}, 对应的相似度为 {1}, 越接近0越相似".format(sorted_score_K[0][0], sorted_score_K[0][1]))


    similar = sorted_score_B[1][0]
    output = cv2.imread(similar)
    output = cv2.resize(output, (400, 300))
    cv2.imshow('search_result', output)
    cv2.waitKey()