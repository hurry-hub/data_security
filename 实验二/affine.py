#!/usr/bin/env python
# coding=utf-8

"""
@author: Richard Huang
@license: WHU
@contact: 2539444133@qq.com
@file: affine.py
@date: 22/03/31 14:01
@desc: 
"""
import numpy as np
import cv2 as cv

# 读取彩色图片为灰度图
img = cv.imread('image/fujisan.jpg')                        # 读取图片，目前为RGB图片
img = cv.resize(img, (576, 402))                            # 缩放图片
rows, cols, channels = img.shape                            # 获取图片参数

# 平移
M = np.float32([[1, 0, 100], [0, 1, 50]])
dst1 = cv.warpAffine(img, M, (cols, rows))

# 旋转
M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 90, 1)
dst2 = cv.warpAffine(img, M, (cols, rows))

# 仿射变换
pts1_ori = [(50, 50), (200, 50), (50, 200)]                 # 原图片上选择三点
pts2_ori = [(10, 100), (200, 50), (100, 250)]               # 仿射图片上选三点
pts1 = np.float32(pts1_ori)
pts2 = np.float32(pts2_ori)
M = cv.getAffineTransform(pts1, pts2)                       # 获得仿射矩阵
dst3 = cv.warpAffine(img, M, (cols, rows))                  # 实现仿射

# 在仿射前后图中标出三点
for point in pts1_ori:
    cv.circle(img, point, 2, (0, 0, 255), 3)
for point in pts2_ori:
    cv.circle(dst3, point, 2, (0, 0, 255), 3)

# 绘制点之间的连线
line_maps = {0: 2, 1: 0, 2: 1}
for i in line_maps.keys():
    start_point = pts1_ori[i]
    end_point = pts1_ori[line_maps[i]]
    cv.line(img, start_point, end_point, (0, 255, 0), 2)
for i in line_maps.keys():
    start_point = pts2_ori[i]
    end_point = pts2_ori[line_maps[i]]
    cv.line(dst3, start_point, end_point, (0, 255, 0), 2)
result = np.hstack((img, dst3))
cv.imshow('compare', result)
cv.imshow('rotation', dst2)
cv.imshow('translation', dst1)
cv.waitKey()
