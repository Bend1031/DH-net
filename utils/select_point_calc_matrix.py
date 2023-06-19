import os
import sys

import cv2
import numpy as np
import scipy.io as sio
from pylab import *

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
from lib.rootpath import rootPath

# 手动选取配准点进行配准融合
srcpoint = []
destpoint = []
sourcepoint = []
targetpoint = []


# 显示图像
def viewImage(image):
    cv2.namedWindow("Display", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Display", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 图像1的点击事件
def click_event_srcimage1(event, x, y, flags, params):
    global srcpoint
    if event == cv2.EVENT_LBUTTONDOWN:
        srcpoint.append((x, y))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image1, str(x) + "," + str(y), (x, y), font, 0.5, (255, 0, 0), 1)
        cv2.imshow("Base Image", image1)


# 图像2的点击事件
def click_event_dstimage2(event, x, y, flags, params):
    global destpoint
    if event == cv2.EVENT_LBUTTONDOWN:
        destpoint.append((x, y))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image2, str(x) + "," + str(y), (x, y), font, 0.5, (255, 0, 0), 1)
        cv2.imshow("Target Image", image2)


def homography_manual():
    global sourcepoint, targetpoint
    sourcepoint = np.array(srcpoint)
    targetpoint = np.array(destpoint)
    # h为仿射矩阵
    h, status = cv2.findHomography(sourcepoint, targetpoint)
    # save h as mat

    sio.savemat("h.mat", {"h": h})

    print(h)
    image_output = cv2.warpPerspective(image1, h, (image2.shape[1], image2.shape[0]))
    viewImage(image_output)
    rate = 0.5
    # 两张图像重合显示
    overlapping = cv2.addWeighted(image2, rate, image_output, 1 - rate, 0)
    viewImage(overlapping)


if __name__ == "__main__":
    # 读取图像
    # %% load image
    imgfile1 = rootPath / "datasets/test_dataset/4sar-optical/pair2-1.jpg"
    imgfile2 = rootPath / "datasets/test_dataset/4sar-optical/pair2-2.jpg"
    image1 = cv2.imread(str(imgfile1))
    image2 = cv2.imread(str(imgfile2))
    # [image1, image2, I1rgb, I2rgb, path1, path2] = read.readImage()
    cv2.namedWindow("Base Image")
    # cv2.resizeWindow('Base Image', 640, 512)  # 自己设定窗口图片的大小
    cv2.imshow("Base Image", image1)
    cv2.setMouseCallback("Base Image", click_event_srcimage1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.namedWindow("Target Image")
    # cv2.resizeWindow('Target Image', 640, 512)  # 自己设定窗口图片的大小
    cv2.imshow("Target Image", image2)
    cv2.setMouseCallback("Target Image", click_event_dstimage2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    homography_manual()
