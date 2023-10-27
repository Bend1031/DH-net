import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, transform

from lib import plotmatch
from lib.cnn_feature import cnn_feature_extract

# time count
start = time.perf_counter()

_RESIDUAL_THRESHOLD = 3
# %% load image
imgfile1 = "datasets/SOPatch/WHU-SEN-City/val/opt/d10011.png"
imgfile2 = "datasets/SOPatch/WHU-SEN-City/val/sar/d10011.png"
# imgfile1 = "test_imgs/whu/NH49E001013_5_opt.tif"
# imgfile2 = "test_imgs/whu/NH49E001013_5_sar.tif"

start = time.perf_counter()

# read left image
# int8 ndarray (H, W, C) C=3
# image1 = imageio.v3.imread(imgfile1)
# image2 = imageio.v3.imread(imgfile2)
image1 = cv2.imread(imgfile1)
image2 = cv2.imread(imgfile2)

print("read image time is %6.3f" % (time.perf_counter() - start))

start0 = time.perf_counter()
# 利用CNN提特征 keypoints, scores, descriptors
# keypoints: (N, 3) ndarray of float32 (x, y, scale)? scale=4
# scores: (N,) ndarray of float32
# descriptors: (N, 512) ndarray of float32
kps_left, sco_left, des_left = cnn_feature_extract(image1, nfeatures=-1)
kps_right, sco_right, des_right = cnn_feature_extract(image2, nfeatures=-1)

print(
    "Feature_extract time is %6.3f, left: %6.3f,right %6.3f"
    % ((time.perf_counter() - start), len(kps_left), len(kps_right))
)

start = time.perf_counter()

# %% Flann特征匹配
# 优点：批量特征匹配时，FLANN速度快；
# 缺点：由于使用的是邻近近似值，所以精度较差
# Index_params字典：匹配算法KDTREE,LSH;
# Search_parames字典:指定KDTREE算法中遍历树的次数；
FLANN_INDEX_KDTREE = 1  #
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  #
search_params = dict(checks=40)  #
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_left, des_right, k=2)  # 前2个最相近的特征向量
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
min_dist = 1000
max_dist = 0
disdif_avg = 0
# 统计平均距离差
for m, n in matches:
    disdif_avg += n.distance - m.distance
disdif_avg = disdif_avg / len(matches)

for m, n in matches:
    # 自适应阈值
    if n.distance > m.distance + disdif_avg:
        goodMatch.append(m)
        p2 = cv2.KeyPoint(kps_right[m.trainIdx][0], kps_right[m.trainIdx][1], 1)
        p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
        locations_1_to_use.append([p1.pt[0], p1.pt[1]])
        locations_2_to_use.append([p2.pt[0], p2.pt[1]])
# goodMatch = sorted(goodMatch, key=lambda x: x.distance)
print("match num is %d" % len(goodMatch))
locations_1_to_use = np.array(locations_1_to_use)
locations_2_to_use = np.array(locations_2_to_use)

# %% Perform geometric verification using RANSAC.
ransac_model, inliers = measure.ransac(
    (locations_1_to_use, locations_2_to_use),
    transform.AffineTransform,
    min_samples=4,
    residual_threshold=_RESIDUAL_THRESHOLD,
    max_trials=1000,
)
Fm, inliers = cv2.findFundamentalMat(
    locations_1_to_use, locations_2_to_use, cv2.USAC_MAGSAC, 0.5, 0.999, 100000
)

print("Found %d inliers" % sum(inliers))

inlier_idxs = np.nonzero(inliers)[0]
# 最终匹配结果
matches = np.column_stack((inlier_idxs, inlier_idxs))
print("whole time is %6.3f" % (time.perf_counter() - start0))


# Visualize correspondences, and save to file.
# 1 绘制匹配连线
plt.rcParams["savefig.dpi"] = 300  # 图片像素
plt.rcParams["figure.dpi"] = 300  # 分辨率
plt.rcParams["figure.figsize"] = (4.0, 3.0)  # 设置figure_size尺寸
_, ax = plt.subplots()
plotmatch.plot_matches(
    ax,
    image1,
    image2,
    locations_1_to_use,
    locations_2_to_use,
    np.column_stack((inlier_idxs, inlier_idxs)),
    plot_matche_points=False,
    matchline=True,
    matchlinewidth=0.3,
)
ax.axis("off")
ax.set_title("")
plt.show()

image3 = transform.warp(image1, ransac_model.params)
# 2 绘制拼接图
plt.rcParams["savefig.dpi"] = 300  # 图片像素
plt.rcParams["figure.dpi"] = 300  # 分辨率
plt.rcParams["figure.figsize"] = (4.0, 3.0)  # 设置figure _size尺寸
plt.subplot(1, 2, 1)
plt.imshow(image1)
plt.subplot(1, 2, 2)
plt.imshow(image3)
plt.show()
