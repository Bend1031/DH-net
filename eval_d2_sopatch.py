"""验证d2原始权重+mnn+magsac(速度快)在sopatch_whu数据集test文件夹下的mRMSE
"""

import glob

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lib.model_test import D2Net
from lib.pyramid import process_multiscale
from lib.utils import BruteForce, flann_match, magsac, pix2pix_RMSE, preprocess_image


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


# Creating CNN model
use_cuda = torch.cuda.is_available()
model = D2Net(model_file="models/d2_tf.pth", use_cuda=use_cuda)
# model = D2Net(model_file="models/d2_tf_no_phototourism.pth", use_cuda=use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
multiscale = False
max_edge = 2500
max_sum_edges = 5000


def main(cnn_feature_extract, multiscale, imgfile1, imgfile2):
    image1 = cv2.imread(imgfile1, 0)
    image2 = cv2.imread(imgfile2, 0)

    kps_left, sco_left, des_left = cnn_feature_extract(image1, multiscale, nfeatures=-1)
    kps_right, sco_right, des_right = cnn_feature_extract(
        image2, multiscale, nfeatures=-1
    )

    # %%flann快速最近邻匹配
    # 二者数量需要相等
    # locations_1_to_use, locations_2_to_use = flann_match(
    #     kps_left, kps_right, des_left, des_right
    # )
    locations_1_to_use, locations_2_to_use = BruteForce(
        kps_left, kps_right, des_left, des_right
    )

    if len(locations_1_to_use) <= 4 or len(locations_2_to_use) <= 4:
        print("flann后点数过少")
        return 0, 0, 0

    # %% Perform geometric verification using magsac.
    H, inliers = magsac(
        locations_1_to_use,
        locations_2_to_use,
    )

    inlier_idxs = np.nonzero(inliers)[0]
    if len(inlier_idxs) < 4:
        print("magsac离散点去除后点数过少")
        return 0, 0, 0

    # %% 计算最终匹配结果
    corr_match1 = locations_1_to_use[inlier_idxs]
    corr_match2 = locations_2_to_use[inlier_idxs]

    RMSE, NCM, CMR, bool_list = pix2pix_RMSE(corr_match1, corr_match2)

    # %%绘图

    # display = evaluation_utils.draw_match(
    #     image1,
    #     image2,
    #     corr_match1,
    #     corr_match2,
    #     inlier=bool_list,
    # )
    # path_name = "/".join(list(PurePath(imgfile1).parts[1:-2]))
    # imgfile_name = PurePath(imgfile1).name
    # img_path = rootPath.joinpath("imgs", path_name, f"d2_flann_magsac_{imgfile_name}")

    # directory = Path(img_path)
    # if not directory.exists():
    #     # 如果目录不存在，则使用mkdir()方法创建目录
    #     directory.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite(str(img_path), display)

    return (NCM, CMR, RMSE)


# datasets/SOPatch/SEN1-2
# imgfiles1 = glob.glob(r"datasets/SOPatch/OSdataset/test/opt/*.png")
# imgfiles2 = glob.glob(r"datasets/SOPatch/OSdataset/test/sar/*.png")
# imgfiles1 = glob.glob(r"datasets/SOPatch/WHU-SEN-City/test/opt/*.png")
# imgfiles2 = glob.glob(r"datasets/SOPatch/WHU-SEN-City/test/sar/*.png")
imgfiles1 = glob.glob(r"datasets/SOPatch/SEN1-2/test/opt/*.png")
imgfiles2 = glob.glob(r"datasets/SOPatch/SEN1-2/test/sar/*.png")

mRMSE = []
mNCM = []
mCMR = []
success_num = 0


for i in tqdm(range(len(imgfiles1))):
    imgfile1 = imgfiles1[i]
    imgfile2 = imgfiles2[i]
    ncm, cmr, rmse = main(cnn_feature_extract, multiscale, imgfile1, imgfile2)
    if ncm >= 5:
        success_num += 1
    mNCM.append(ncm)
    mCMR.append(cmr)
    mRMSE.append(rmse)
# 去除0值
# print("success_rate:", success_num / len(imgfiles1))
mNCM = [x for x in mNCM if x != 0]
mCMR = [x for x in mCMR if x != 0]
mRMSE = [x for x in mRMSE if x != 0]

# 数据分析
print(f"总数据量为{len(imgfiles1)}")
print(f"匹配成功数为{success_num}")
print(f"成功率为{success_num / len(imgfiles1)}")
print(f"NCM均值为{np.mean(mNCM)}")
print(f"CMR均值为{np.mean(mCMR)}")
print(f"RMSE均值为{np.mean(mRMSE)}")
