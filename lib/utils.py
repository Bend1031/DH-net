import argparse
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from lib.exceptions import EmptyTensorError


def preprocess_image(image, preprocessing=None):
    image = image.astype(np.float32)
    image = np.transpose(image, [2, 0, 1])  # [c, h, w]
    if preprocessing is None:
        pass
    elif preprocessing == "caffe":
        # RGB -> BGR
        image = image[::-1, :, :]
        # Zero-center by mean pixel
        mean = np.array([103.939, 116.779, 123.68])
        image = image - mean.reshape([3, 1, 1])
    elif preprocessing == "torch":
        image /= 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean.reshape([3, 1, 1])) / std.reshape([3, 1, 1])
    else:
        raise ValueError("Unknown preprocessing parameter.")
    return image


def imshow_image(image, preprocessing=None):
    if preprocessing is None:
        pass
    elif preprocessing == "caffe":
        mean = np.array([103.939, 116.779, 123.68])
        image = image + mean.reshape([3, 1, 1])
        # RGB -> BGR
        image = image[::-1, :, :]
    elif preprocessing == "torch":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std.reshape([3, 1, 1]) + mean.reshape([3, 1, 1])
        image *= 255.0
    else:
        raise ValueError("Unknown preprocessing parameter.")
    image = np.transpose(image, [1, 2, 0])
    image = np.round(image).astype(np.uint8)
    return image


def grid_positions(h, w, device, matrix=False):
    lines = torch.arange(0, h, device=device).view(-1, 1).float().repeat(1, w)
    columns = torch.arange(0, w, device=device).view(1, -1).float().repeat(h, 1)
    if matrix:
        return torch.stack([lines, columns], dim=0)
    else:
        return torch.cat([lines.view(1, -1), columns.view(1, -1)], dim=0)


def upscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = pos * 2 + 0.5
    return pos


def downscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = (pos - 0.5) / 2
    return pos


def interpolate_dense_features(pos, dense_features, return_corners=False):
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)

    _, h, w = dense_features.size()

    i = pos[0, :]
    j = pos[1, :]

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right),
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    descriptors = (
        w_top_left * dense_features[:, i_top_left, j_top_left]
        + w_top_right * dense_features[:, i_top_right, j_top_right]
        + w_bottom_left * dense_features[:, i_bottom_left, j_bottom_left]
        + w_bottom_right * dense_features[:, i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    if not return_corners:
        return [descriptors, pos, ids]
    else:
        corners = torch.stack(
            [
                torch.stack([i_top_left, j_top_left], dim=0),
                torch.stack([i_top_right, j_top_right], dim=0),
                torch.stack([i_bottom_left, j_bottom_left], dim=0),
                torch.stack([i_bottom_right, j_bottom_right], dim=0),
            ],
            dim=0,
        )
        return [descriptors, pos, ids, corners]


def savefig(filepath, fig=None, dpi=None):
    # TomNorway - https://stackoverflow.com/a/53516034
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    for ax in fig.axes:
        ax.axis("off")
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    fig.savefig(filepath, pad_inches=0, bbox_inches="tight", dpi=dpi)


def parse_args():
    """Argument parsing"""
    parser = argparse.ArgumentParser(description="Training script")

    parser.add_argument(
        "--dataset_path", type=str, required=False, help="path to the dataset"
    )

    parser.add_argument(
        "--preprocessing",
        type=str,
        default="torch",
        help="image preprocessing (caffe or torch)",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="weights/d2/d2_tf.pth",
        help="path to the full model",
    )

    parser.add_argument(
        "--num_epochs", type=int, default=1, help="number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="number of workers for data loading"
    )

    parser.add_argument(
        "--use_validation",
        dest="use_validation",
        action="store_true",
        help="use the validation split",
    )
    parser.set_defaults(use_validation=False)

    parser.add_argument(
        "--log_interval", type=int, default=250, help="loss logging interval"
    )

    parser.add_argument(
        "--log_file", type=str, default="log.txt", help="loss logging file"
    )

    parser.add_argument(
        "--plot", dest="plot", action="store_true", help="plot training pairs"
    )
    parser.set_defaults(plot=False)

    parser.add_argument(
        "--checkpoint_directory",
        type=str,
        default="checkpoints",
        help="directory for training checkpoints",
    )
    parser.add_argument(
        "--checkpoint_prefix",
        type=str,
        default="d2",
        help="prefix for training checkpoints",
    )

    return parser.parse_args()


def pca(X, k):
    """
    X: 数据矩阵，每一行表示一个样本，每一列表示一个特征
    k: 降维后的维数
    """
    pca = PCA(n_components=k)
    X_new = pca.fit_transform(X)

    alpha = 0
    X_norm = np.linalg.norm(X_new, axis=0)
    X_new = X_new / (X_norm + alpha)
    return X_new


def flann_match(kps_left, kps_right, des_left, des_right, ratio_threshold=0.99):
    """
    使用FLANN算法进行特征匹配

    参数：
    kps_left (list): 左图中的关键点列表
    kps_right (list): 右图中的关键点列表
    des_left (numpy.ndarray): 左图中的特征描述子
    des_right (numpy.ndarray): 右图中的特征描述子
    ratio_threshold (float): 用于筛选匹配的阈值，默认为0.99

    返回：
    locations_1_to_use (numpy.ndarray): 用于匹配的左图关键点位置
    locations_2_to_use (numpy.ndarray): 用于匹配的右图关键点位置
    """

    # match_start_time = time.time()

    # 设置FLANN参数
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=40)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 使用默认设置:效果很差
    # flann = cv2.FlannBasedMatcher()

    # 使用FLANN算法进行匹配
    matches = flann.knnMatch(des_left, des_right, k=2)

    goodMatch = []
    locations_1_to_use = []
    locations_2_to_use = []

    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            goodMatch.append(m)
            p2 = cv2.KeyPoint(kps_right[m.trainIdx][0], kps_right[m.trainIdx][1], 1)
            p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
            locations_1_to_use.append([p1.pt[0], p1.pt[1]])
            locations_2_to_use.append([p2.pt[0], p2.pt[1]])

    # match_end_time = time.time()

    # 将关键点位置转换为NumPy数组
    locations_1_to_use = np.array(locations_1_to_use)
    locations_2_to_use = np.array(locations_2_to_use)

    return locations_1_to_use, locations_2_to_use


def BruteForce(kps_left, kps_right, des_left, des_right):
    # 默认 cv2.NORM_L2
    bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des_left, des_right, k=2)
    matches = bf.match(des_left, des_right)
    # goodMatch = []
    locations_1_to_use = []
    locations_2_to_use = []

    # for m, n in matches:
    #     if m.distance < ratio_threshold * n.distance:
    #         goodMatch.append(m)
    #         p2 = cv2.KeyPoint(kps_right[m.trainIdx][0], kps_right[m.trainIdx][1], 1)
    #         p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
    #         locations_1_to_use.append([p1.pt[0], p1.pt[1]])
    #         locations_2_to_use.append([p2.pt[0], p2.pt[1]])

    for m in matches:
        # goodMatch.append(m)
        p2 = cv2.KeyPoint(kps_right[m.trainIdx][0], kps_right[m.trainIdx][1], 1)
        p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
        locations_1_to_use.append([p1.pt[0], p1.pt[1]])
        locations_2_to_use.append([p2.pt[0], p2.pt[1]])

    # match_end_time = time.time()

    # 将关键点位置转换为NumPy数组
    locations_1_to_use = np.array(locations_1_to_use)
    locations_2_to_use = np.array(locations_2_to_use)

    return locations_1_to_use, locations_2_to_use


def calcRMSE(src_pts, dst_pts, M):
    """
    计算RMSE (Root Mean Squared Error) 误差。

    参数:
    src_pts (ndarray): 源点坐标，形状为 (N, 2)
    dst_pts (ndarray): 目标点坐标，形状为 (N, 2)
    M (ndarray): 变换矩阵，形状为 (3, 3)

    返回:
    rmse (float): 计算得到的RMSE误差
    """
    # 求残差
    sum_H = 0  # 残差和
    num = 0  # 参与统计的总个数
    for i, j in zip(src_pts, dst_pts):
        # 将源点通过变换M变换到目标点
        i = np.array([i[0], i[1], 1]).reshape(3, 1)
        j = np.array([j[0], j[1], 1]).reshape(3, 1)
        H_i = np.dot(M, i)
        H_i = H_i / H_i[2]
        # 计算残差
        diff = H_i[:2] - j[:2]
        sum_H += np.sum(diff**2)
        num += 1
    # 计算RMSE
    rmse = np.sqrt(sum_H / num)
    # print(f"RMSE:{rmse:.3f}")
    return rmse


def calc_pix_RMSE(src_pts, M):
    """像素对应,按此计算并非真值的RMSE"""
    # 求残差
    sum_H = 0  # 残差和
    num = 0  # 参与统计的总个数
    for i in src_pts:
        # 将源点通过变换M变换到目标点
        i = np.array([i[0], i[1], 1]).reshape(3, 1)
        # j = np.array([j[0], j[1], 1]).reshape(3, 1)
        H_i = np.dot(M, i)
        H_i = H_i / H_i[2]
        # 计算残差
        diff = H_i[:2] - i[:2]
        sum_H += np.sum(diff**2)
        num += 1
    # 计算RMSE
    rmse = np.sqrt(sum_H / num)
    print(f"RMSE:{rmse:.3f}")
    return rmse


def pix2pix_RMSE(src_pts, dst_pts, threshold=3):
    """
    计算两组对应点之间的均方根误差(RMSE)。

    Parameters:
        src_pts (numpy.ndarray): 第一组点的坐标,表示为一个n*2的NumPy数组。
        dst_pts (numpy.ndarray): 第二组点的坐标,表示为一个n*2的NumPy数组。应与src_pts具有相同数量的点。

    Returns:
        float: 两组对应点之间的均方根误差(RMSE)。

    Example:
        src_pts = np.array([[x1, y1], [x2, y2], ...])  # 第一组点
        dst_pts = np.array([[x1, y1], [x2, y2], ...])  # 第二组点
        rmse = pix2pix_RMSE(src_pts, dst_pts)
        print("均方根误差(RMSE):", rmse)
    """
    # 计算两组对应点的距离并求平均rmse
    # import numpy as np

    # 输入两组点，分别表示为两个n*2的NumPy数组
    points1 = src_pts  # 第一组点
    points2 = dst_pts  # 第二组点

    # 计算对应点之间的距离
    distances = np.linalg.norm(points1 - points2, axis=1)

    err = {}
    for thr in range(1, 11):
        err[thr] = np.mean(distances <= thr)

    # 根据阈值剔除大于阈值的点
    filter_idx = np.where(distances < threshold)
    NCM = len(filter_idx[0])
    CMR = NCM / len(points1)

    filter_distance = distances[filter_idx]
    bool_list = distances < threshold
    # print(bool_list)

    # 计算均方根误差（RMSE）
    # 所有点都没成功配准
    if filter_distance.size == 0:
        RMSE = 0
    else:
        RMSE = np.sqrt(np.mean(filter_distance**2))
    # RMSE=np.mean(distances**2)

    # print("对应点之间的距离:", filter_distance)
    # print(f"RMSE: {rmse:.2f}")
    return RMSE, NCM, CMR, bool_list, err


def magsac(srcPoints, dstPoints, method=cv2.USAC_MAGSAC, _RESIDUAL_THRESHOLD=3):
    H, inliers = cv2.findHomography(
        srcPoints=srcPoints,
        dstPoints=dstPoints,
        method=method,
        ransacReprojThreshold=_RESIDUAL_THRESHOLD,
        confidence=0.999,
        maxIters=10000,
    )
    return H, inliers


def rotate_image(image_path, angle):
    # 读取图像
    image = cv2.imread(image_path)

    # 计算图像中心点
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    # 构建旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 执行旋转操作
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image
