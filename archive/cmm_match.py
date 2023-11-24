"""CMM方法
"""
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
import scipy.misc
import torch
from PIL import Image
from skimage import measure, transform
from skimage.feature import match_descriptors

from lib import plotmatch
from lib.eval_match import img_align
from lib.model_test import D2Net
from lib.pyramid import process_multiscale
from lib.rootpath import rootPath
from lib.utils import pix2pix_RMSE, preprocess_image
from utils import evaluation_utils


def resize_image_with_pil(image, scale_factor):
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    input_pil_image = Image.fromarray(image.astype("uint8"))
    resized_image = input_pil_image.resize((new_width, new_height), Image.ANTIALIAS)
    return np.array(resized_image).astype("float")


# de-net feature extract function
def cnn_feature_extract(image, scales=[0.25, 0.50, 1.0], nfeatures=1000):
    multiscale = False
    # repeat single channel image to 3 channel
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # Resize image to maximum size.
    resized_image = image

    # 如果最大边大于max_edge，则调整大小
    if max(resized_image.shape) > max_edge:
        scale_factor = max_edge / max(resized_image.shape)
        resized_image = resize_image_with_pil(resized_image, scale_factor)

    # 如果尺寸之和大于max_sum_edges，则调整大小
    if sum(resized_image.shape[:2]) > max_sum_edges:
        scale_factor = max_sum_edges / sum(resized_image.shape[:2])
        resized_image = resize_image_with_pil(resized_image, scale_factor)

    # resize proportion
    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(resized_image, preprocessing="torch")
    with torch.inference_mode():
        # Process image with D2-Net
        if multiscale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32), device=device
                ),
                model,
                scales,
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32), device=device
                ),
                model,
                scales=[1],
            )

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]

    if nfeatures != -1:
        # 根据scores排序
        scores2 = np.array([scores]).T
        res = np.hstack((scores2, keypoints))
        res = res[np.lexsort(-res[:, ::-1].T)]

        res = np.hstack((res, descriptors))
        # 取前几个
        scores = res[0:nfeatures, 0].copy()
        keypoints = res[0:nfeatures, 1:4].copy()
        descriptors = res[0:nfeatures, 4:].copy()
        del res
    return keypoints, scores, descriptors


use_cuda = torch.cuda.is_available()
# Creating CNN model
model = D2Net(model_file="models/d2_tf.pth", use_cuda=use_cuda)
# model = D2Net(model_file="checkpoints/qxslab/qxs.best.pth", use_cuda=use_cuda)
# model = D2Net(model_file="checkpoints/whu_crop/whu.10.pth", use_cuda=use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")

multiscale = False
max_edge = 2500
max_sum_edges = 5000

_RESIDUAL_THRESHOLD = 3
# %% load image
imgfile1 = "datasets/test_dataset/4SARSets/pair8-1.tif"
imgfile2 = "datasets/test_dataset/4SARSets/pair8-2.tif"

# read left image
# int8 ndarray (H, W, C) C=3

image1 = cv2.imread(imgfile1)
image2 = cv2.imread(imgfile2)


kps_left, sco_left, des_left = cnn_feature_extract(image1, nfeatures=-1)
kps_right, sco_right, des_right = cnn_feature_extract(image2, nfeatures=-1)
print("left feature num is %d" % len(kps_left))
print("right feature num is %d" % len(kps_right))


# %% Flann特征匹配
# 优点：批量特征匹配时，FLANN速度快；
# 缺点：由于使用的是邻近近似值，所以精度较差
# Index_params字典：匹配算法KDTREE,LSH;
# Search_parames字典:指定KDTREE算法中遍历树的次数；
match_start_time = time.time()
FLANN_INDEX_KDTREE = 1  #
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  #
search_params = dict(checks=40)  #
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(
    des_left,
    des_right,
    k=2,
)  # 前2个最相近的特征向量
# matches Dmatch object
# 其中DMatch中的内容：
# Distance:描述子之间的距离，值越低越好；
# queryIdx:第一幅图的描述子索引值；
# TrainIdx:第二幅图的描述子索引值；
# imgIdx:第二幅图的索引值；

goodMatch = []
locations_1_to_use = []
locations_2_to_use = []

# 匹配对筛选
# min_dist = 1000
# max_dist = 0
disdif_avg = 0
# 统计平均距离差
for m, n in matches:
    disdif_avg += n.distance - m.distance
disdif_avg = disdif_avg / len(matches)
# print(f"ratio={disdif_avg:.2f}")

for m, n in matches:
    # 自适应阈值
    if n.distance > m.distance + disdif_avg:
        # if m.distance < 0.8 * n.distance:
        goodMatch.append(m)
        p2 = cv2.KeyPoint(kps_right[m.trainIdx][0], kps_right[m.trainIdx][1], 1)
        p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
        locations_1_to_use.append([p1.pt[0], p1.pt[1]])
        locations_2_to_use.append([p2.pt[0], p2.pt[1]])
goodMatch = sorted(goodMatch, key=lambda x: x.distance)
# match_end_time = time.time()
# print("match num is %d" % len(goodMatch))
# print("match time is %6.3f ms" % ((match_end_time - match_start_time) * 1000))
locations_1_to_use = np.array(locations_1_to_use)
locations_2_to_use = np.array(locations_2_to_use)

# %% Perform geometric verification using RANSAC.

# Fm, inliers = cv2.findFundamentalMat(
#     locations_1_to_use, locations_2_to_use, cv2.USAC_MAGSAC, 3, 0.999, 100000
# )
start_time = time.time()
# H, inliers = measure.ransac(
#     (locations_1_to_use, locations_2_to_use),
#     transform.AffineTransform,
#     min_samples=4,
#     residual_threshold=_RESIDUAL_THRESHOLD,
#     max_trials=100000,
# )
H, inliers = cv2.findHomography(
    srcPoints=locations_1_to_use,
    dstPoints=locations_2_to_use,
    method=cv2.RANSAC,
    ransacReprojThreshold=_RESIDUAL_THRESHOLD,
    confidence=0.999,
    maxIters=10000,
)
end_time = time.time()
print("ransac time is %6.3f ms" % ((end_time - start_time) * 1000))
# H, inliers = cv2.findHomography(
#     srcPoints=locations_1_to_use,
#     dstPoints=locations_2_to_use,
#     method=cv2.USAC_MAGSAC,
#     ransacReprojThreshold=3,
#     confidence=0.999,
#     maxIters=100000,
# )

print("Found %d inliers" % sum(inliers))


inlier_idxs = np.nonzero(inliers)[0]
corr_match1 = locations_1_to_use[inlier_idxs]
corr_match2 = locations_2_to_use[inlier_idxs]
rmse, NCM, CMR, bool_list = pix2pix_RMSE(corr_match1, corr_match2)
# CMR = NCM / len(corr_match1)
print(f"RMSE: {rmse:.2f}")
print(f"CMR: {CMR:.2f}")
# 最终匹配结果
matches = np.column_stack((inlier_idxs, inlier_idxs))

# %% evaluation
# show align image
# draw points
# dis_points_1 = evaluation_utils.draw_points(image1, kps_left)
# dis_points_2 = evaluation_utils.draw_points(image2, kps_right)

# visualize match
display = evaluation_utils.draw_match(
    image1,
    image2,
    locations_1_to_use[inlier_idxs],
    locations_2_to_use[inlier_idxs],
    inlier=bool_list,
)
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow(
    "test.png",
    display,
)
cv2.waitKey(0)
# img_align(image1, image2, H)
