# ------------------------------------------------------------------
# 作者：柯大锋，胡大鹏 日期：2021.09.17
# 描述：通过在图片上手动选区，计算区域内晶体的面积占比
# 适用范围：默认灰度图中晶体区域的灰度大于基底，若晶体太厚或太暗可能不适用，
#           请比较二值图与原图是否匹配
# 安装依赖：pip install opencv-python matplotlib numpy
# ------------------------------------------------------------------

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


def calCoverage(img0, blockSize=51, display=True):
    '''
    高斯自适应阈值法将图片处理成灰度二值图，对黑白像素进行统计
    img0: Image Array
    blockSize: 邻域块大小
    display: 是否显示原图、灰度图和二值图
    '''

    # #亮度平衡
    # YUV = cv.cvtColor(img0, cv.COLOR_BGR2YUV)
    # YUV[:, :, 0] = cv.equalizeHist(YUV[:, :, 0])
    # equalized_color = cv.cvtColor(YUV, cv.COLOR_YUV2BGR)
    # cv.imshow("equalized_color", equalized_color)

    # 转化成灰度图
    gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    size = h * w
    # 二值化
    bin = gray.copy()
    dw = int(w / 6)
    for col in range(0, w, dw):
        roi = bin[:, col : col + dw]
        binary = cv.adaptiveThreshold(
            roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, 6
        )
        # ret, binary = cv.threshold(roi, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        bin[:, col : col + dw] = binary
    # print(ret)
    # 计算沟道的占比
    count = 0
    for i in range(h):
        for j in range(w):
            if bin[i, j] == 255:
                count += 1
    content = count / size
    print('沟道内晶体面积比：%.2f' % (content * 100) + '%')

    # 显示图片
    if display:
        cv.imshow('img0', img0)
        cv.imshow('gray', gray)
        cv.imshow('binary', bin)
        cv.waitKey()
        cv.destroyAllWindows()

    return content


if __name__ == '__main__':
    img = plt.imread(input('文件名：'))
    while True:
        plt.imshow(img)
        print('选取计算区域的左上角和右下角')
        pts = []
        while len(pts) < 2:
            pts = plt.ginput(2, timeout=-1)
        img0 = img[int(pts[0][1]) : int(pts[1][1]), int(pts[0][0]) : int(pts[1][0]), :]
        calCoverage(img0)
