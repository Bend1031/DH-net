"""d2原始权重+mnn+magsac(速度快)
"""
import time

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.feature import match_descriptors

from lib.eval_match import img_align
from lib.model_test import D2Net
from lib.pyramid import process_multiscale
from lib.rootpath import rootPath
from lib.utils import calcRMSE, flann, pca, preprocess_image
from utils import evaluation_utils


def resize_image_with_pil(image, scale_factor):
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    input_pil_image = Image.fromarray(image.astype("uint8"))
    resized_image = input_pil_image.resize((new_width, new_height), Image.ANTIALIAS)
    return np.array(resized_image).astype("float")


# de-net feature extract function
def cnn_feature_extract(image, multiscale, scales=[0.25, 0.50, 1.0], nfeatures=1000):
    # multiscale = True
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
    with torch.no_grad():
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


_RESIDUAL_THRESHOLD = 1
# METHOD = cv2.RANSAC
METHOD = cv2.USAC_MAGSAC

# Creating CNN model
use_cuda = torch.cuda.is_available()
model = D2Net(model_file="models/d2_tf.pth", use_cuda=use_cuda)
# model = D2Net(model_file="checkpoints/qxslab256/qxs.best.pth", use_cuda=use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
multiscale = False
max_edge = 2500
max_sum_edges = 5000
# %% load image
imgfile1 = "datasets/test_dataset/4sar-optical/pair1-1.jpg"
imgfile2 = "datasets/test_dataset/4sar-optical/pair1-2.jpg"

# read left image
# int8 ndarray (H, W, C) C=3

image1 = cv2.imread(imgfile1)
image2 = cv2.imread(imgfile2)

feature_extract_start_time = time.time()
kps_left, sco_left, des_left = cnn_feature_extract(image1, multiscale, nfeatures=-1)
kps_right, sco_right, des_right = cnn_feature_extract(image2, multiscale, nfeatures=-1)
feature_extract_end_time = time.time()
print("left feature num is %d" % len(kps_left))
print("right feature num is %d" % len(kps_right))
print(
    "feature_extract took %.2f ms."
    % ((feature_extract_end_time - feature_extract_start_time) * 1000)
)
# %%PCA
# des_left = pca(des_left, 256)
# des_right = pca(des_right, 256)
# %% 特征匹配
# match: (M, 2) array, where M is the number of matches. For each row, the first
# element is the index of the feature in the first image and the second element
# is the index of the feature in the second image.

# match_start_time = time.time()
# matches = match_descriptors(des_left, des_right, cross_check=True)
# match_end_time = time.time()
# locations_1_to_use = kps_left[matches[:, 0], :2]
# locations_2_to_use = kps_right[matches[:, 1], :2]

# print("Number of raw matches: %d." % matches.shape[0])
# print(f"match_descriptors took {((match_end_time - match_start_time) * 1000):.2f} ms.")
# 二者数量需要相等
locations_1_to_use, locations_2_to_use = flann(kps_left, kps_right, des_left, des_right)

# %% Perform geometric verification using RANSAC.
start_time = time.time()
H, inliers = cv2.findHomography(
    srcPoints=locations_1_to_use,
    dstPoints=locations_2_to_use,
    method=METHOD,
    ransacReprojThreshold=_RESIDUAL_THRESHOLD,
    confidence=0.999,
    maxIters=10000,
)
end_time = time.time()
print("Found %d inliers" % sum(inliers))
print(f"{METHOD} took {((end_time - start_time) * 1000):.2f} ms.")


inlier_idxs = np.nonzero(inliers)[0]
if len(inlier_idxs) < 4:
    print("Can't find enough inliers")
    exit(0)
# 最终匹配结果
# matches = np.column_stack((inlier_idxs, inlier_idxs))
corr_match1 = locations_1_to_use[inlier_idxs]
corr_match2 = locations_2_to_use[inlier_idxs]
calcRMSE(corr_match1, corr_match2, H)
# %% evaluation
# visualize match
display = evaluation_utils.draw_match(
    image1,
    image2,
    corr_match1,
    corr_match2,
    # inlier=inliers,
)

# 显示图片
cv2.imshow(
    "result.png",
    display,
)

# 保存图片
cv2.imwrite("d2_flann_magsac.png", display)
img_align(image1, image2, H)
