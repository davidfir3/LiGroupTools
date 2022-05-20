# ------------------------------------------------------------------
# 作者：柯大锋，胡大鹏 日期：2021.09.17
# 描述：通过在图片上手动选区，计算区域内晶体的面积占比
# 适用范围：默认灰度图中晶体区域的灰度大于基底，若晶体太厚或太暗可能不适用，
#           请比较二值图与原图是否匹配
# 安装依赖：pip install opencv-python matplotlib numpy
# ------------------------------------------------------------------

import cv2 as cv
import numpy as np


def gray2bin(gray, C=6, blockSize=51):
    '''
    Convert grayscale to binary images
    gray: grayscale image
    C: Threshold = Mean - C
    blockSize
    '''
    bin = gray.copy()
    h, w = gray.shape[:2]
    dw = int(w / 6)
    for col in range(0, w, dw):
        roi = gray[:, col: col + dw]
        binary = cv.adaptiveThreshold(
            roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, C
        )
        bin[:, col: col + dw] = binary
    return bin


def detectNW(bin, sin_theta=0.4, min_arcL=100):
    '''
    Detect and circle up nanowires with desired length and orientation
    sin_theta: angle between line vector and the horizon
    min_arcL
    '''
    contours, hierarchy = cv.findContours(
        bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    frames = []
    for contour in contours:
        vx, vy, _, _ = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)
        arcL = cv.arcLength(contour, False)
        # print(arcL)
        if arcL > min_arcL and abs(vy) < sin_theta:
            # rect = cv.minAreaRect(contour)
            # frame = cv.boxPoints(rect)
            # frame = np.int0(frame)
            frame = cv.minEnclosingCircle(contour)
            frames.append(frame)
    return frames


def judgeMorph(img):
    '''
    高斯自适应阈值法将图片处理成灰度二值图，对黑白像素进行统计
    img: Image Array
    '''
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.bilateralFilter(gray, 10, 75, 75)

    binNW = gray2bin(blur, -20)
    binXTAL = gray2bin(blur, 2)
    binXTAL = cv.cvtColor(binXTAL, cv.COLOR_GRAY2BGR)

    frames = detectNW(binNW, 0.2)
    print(len(frames))
    # cv.drawContours(img, frames, -1, 255, 10)
    # cv.drawContours(binXTAL, frames, -1, 255, 10)
    for (x, y), radius in frames:
        cv.circle(img, (int(x), int(y)), int(radius), 255, 10)
    p = []
    for (x, y), radius in frames:
        crop_orig = binXTAL[int(y-radius):int(y+radius), int(x-radius):int(x+radius), 0].copy()
        crop_mirrory = crop_orig[::-1, :].copy()
        diff = crop_orig ^ crop_mirrory
        p.append(diff.sum()/255/diff.size)
        cv.circle(binXTAL, (int(x), int(y)), int(radius), 255, 10)  
    print(sum(p)/len(p))

    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img', img)
    cv.namedWindow('XTAL', cv.WINDOW_NORMAL)
    cv.imshow('XTAL', binXTAL)
    cv.namedWindow("NW", cv.WINDOW_NORMAL)
    cv.imshow('NW', binNW)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    img = cv.imread('example4.jpg')
    judgeMorph(img)
