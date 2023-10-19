# ------------------------------------------------------------------
# 作者：柯大锋，胡大鹏 日期：2021.09.17
# 描述：通过在图片上手动选区，计算区域内晶体的面积占比
# 适用范围：默认灰度图中晶体区域的灰度大于基底，若晶体太厚或太暗可能不适用，
#           请比较二值图与原图是否匹配
# 安装依赖：pip install opencv-python matplotlib numpy
# ------------------------------------------------------------------

import cv2 as cv
import numpy as np
from regex import W
import os


def gray2bin(img, C=6, blockSize=201):
    '''
    Convert grayscale to binary images
    gray: grayscale image
    C: Threshold = Mean - C
    blockSize
    '''
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.bilateralFilter(gray, 10, 75, 75)
    bin = blur.copy()
    h, w = blur.shape[:2]
    dw = int(w / 1)
    for col in range(0, w, dw):
        roi = blur[:, col: col + dw]
        binary = cv.adaptiveThreshold(
            roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, C
        )
        bin[:, col: col + dw] = binary
    return bin


def detectNW(bin, sin_theta=0.2, min_arcL=200):
    '''
    Detect and rect up nanowires with desired length and orientation
    sin_theta: angle between line vector and the horizon
    min_arcL
    '''
    contours, hierarchy = cv.findContours(
        bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rects = []
    for contour in contours:
        vx, vy, x0, y0 = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)
        arcL = cv.arcLength(contour, False)
        # print(arcL)
        if arcL > min_arcL and abs(vy) < sin_theta:
            # rect = cv.minAreaRect(contour)
            # rect = cv.boxPoints(rect)
            # rect = np.int0(rect)
            x,y,w,h = cv.boundingRect(contour)
            rects.append([x,y,w,h])
    return rects


def drawBox(img, bin, rects, threshold=0.8):
    p = []
    count = 0
    for x,y,w,h in rects:
        x0, x1, y0, y1 = x, x+w, y, y+h
        if x0<0 or y0<0 or x1>img.shape[1] or y1>img.shape[0]:
            continue
        crop_orig = bin[y0:y1, x0:x1, 0].copy()
        crop_orig //= 255
        if crop_orig.sum()/crop_orig.size < 0.25:
            continue
        crop_mirrory = crop_orig[::-1, :].copy()
        diff = crop_orig ^ crop_mirrory
        p.append(1-diff.sum()/diff.size)
        color = (255, 0, 0) if p[-1] > threshold else (0, 0, 255)
        if p[-1] < 0.61 and p[-1]>0.60:
            cv.rectangle(img, (x0,y0), (x1,y1), [0, 0, 255], 5)
            cv.rectangle(bin, (x0,y0), (x1,y1), [0, 0, 255], 5)
            cv.imwrite('crop.jpg', crop_orig*255)
            cv.imwrite('crop_diff.jpg', diff*255)
        cv.rectangle(img, (x0,y0), (x1,y1), color, 5)
        cv.putText(img, str(count), (x0, y0), fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale = 10, color = (255,255,255))
        count += 1
        # cv.rectangle(bin, (x0,y0), (x1,y1), color, 5)
    return p


def judgeMorph(img, img_p):
    '''
    高斯自适应阈值法将图片处理成灰度二值图，对黑白像素进行统计
    img: Image Array
    '''
    
    binNW = gray2bin(img_p, -15)

    binXTAL = gray2bin(img, 5, 2001)
    binXTAL = cv.cvtColor(binXTAL, cv.COLOR_GRAY2BGR)

    rects = detectNW(binNW, 0.25, 200)
    p = drawBox(img, binXTAL, rects)
    print((p), len(p))
    if p: print(sum(p)/len(p))

    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img', img)
    cv.namedWindow('img_p', cv.WINDOW_NORMAL)
    cv.imshow('img_p', img_p)
    cv.namedWindow('XTAL', cv.WINDOW_NORMAL)
    cv.imshow('XTAL', binXTAL)
    cv.namedWindow("NW", cv.WINDOW_NORMAL)
    cv.imshow('NW', binNW)
    cv.waitKey()
    cv.destroyAllWindows()
    cv.imwrite('img.jpg', img)
    cv.imwrite('img_p.jpg', img_p)
    cv.imwrite('binXTAL.jpg', binXTAL)
    cv.imwrite('binNW.jpg', binNW)

    return p


if __name__ == '__main__':
    dir = 'C:/Users/ksf/desktop/KSF_OM/C8-BTBT/20220620_XF90nmPTS_0.4_25_needle/hexane/'
    files = os.listdir(dir)
    for i in ['b']:
        for j in ['1']:
            if i+j+'.jpg' in files:
                img = cv.imread(dir+i+j+'.jpg')[:, -3264:, :]
                print(img.shape)
                img_p = cv.imread(dir+i+j+'p.jpg')[:, -3264:, :]
                p = judgeMorph(img, img_p)
                # with open('heptane.csv', 'a') as f:
                #     [f.write('%.4f\n' % l) for l in p]
