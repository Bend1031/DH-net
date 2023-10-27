import cv2
import h5py
import numpy as np


def normalize_intrinsic(x, K):
    # print(x,K)
    return (x - K[:2, 2]) / np.diag(K)[:2]


def normalize_size(x, size, scale=1):
    size = size.reshape([1, 2])
    norm_fac = size.max()
    return (x - size / 2 + 0.5) / (norm_fac * scale)


def np_skew_symmetric(v):
    zero = np.zeros_like(v[:, 0])
    M = np.stack(
        [
            zero,
            -v[:, 2],
            v[:, 1],
            v[:, 2],
            zero,
            -v[:, 0],
            -v[:, 1],
            v[:, 0],
            zero,
        ],
        axis=1,
    )
    return M


def draw_points(img, points, color=(0, 255, 0), radius=3):
    dp = [(int(points[i, 0]), int(points[i, 1])) for i in range(points.shape[0])]
    for i in range(points.shape[0]):
        cv2.circle(img, dp[i], radius=radius, color=color)
    return img


def draw_match(
    img1,
    img2,
    corr1,
    corr2,
    inlier=[True],
    color=None,
    radius1=1,
    radius2=1,
    resize=None,
):
    """
    在两个图像上绘制匹配的特征点或线段，并返回结果图像。

    参数：
    img1: numpy.ndarray
        第一个输入图像，可以是任意通道数的彩色或灰度图像。
    img2: numpy.ndarray
        第二个输入图像，与img1具有相同的通道数和深度。
    corr1: numpy.ndarray
        包含第一个图像中特征点或线段的坐标的数组。
    corr2: numpy.ndarray
        包含第二个图像中特征点或线段的坐标的数组。应与corr1具有相同的长度。
    inlier: list of bool, 可选 (默认值为 [True])
        一个布尔值列表，指示每个匹配是否为内点。如果提供，应与corr1和corr2具有相同的长度。
    color: list of tuple, 可选
        用于绘制匹配的颜色列表。如果未提供，内点将以绿色表示，外点将以红色表示。
        如果仅提供一个颜色元组，则将所有匹配绘制为该颜色。
    radius1: int, 可选 (默认值为 1)
        第一个图像中特征点的半径（用于绘制特征点）。
    radius2: int, 可选 (默认值为 1)
        第二个图像中特征点的半径（用于绘制特征点）。
    resize: tuple of int, 可选
        一个包含两个整数的元组，表示要缩放图像的目标大小（宽度，高度）。
        如果提供，函数将首先将图像缩放到目标大小，然后绘制匹配。

    返回：
    display: numpy.ndarray
        绘制了匹配的结果图像，其中特征点或线段以给定的颜色表示。

    注意：
    - 如果提供了resize参数，函数将先缩放图像，然后绘制匹配。在这种情况下，corr1和corr2中的坐标也将相应地缩放。
    - 如果提供了color参数，可以根据匹配的内外点状态来绘制不同的颜色。
    - 如果未提供color参数，则内点将以绿色表示，外点将以红色表示。
    """
    if resize is not None:
        scale1, scale2 = [img1.shape[1] / resize[0], img1.shape[0] / resize[1]], [
            img2.shape[1] / resize[0],
            img2.shape[0] / resize[1],
        ]
        img1, img2 = cv2.resize(img1, resize, interpolation=cv2.INTER_AREA), cv2.resize(
            img2, resize, interpolation=cv2.INTER_AREA
        )
        corr1, corr2 = (
            corr1 / np.asarray(scale1)[np.newaxis],
            corr2 / np.asarray(scale2)[np.newaxis],
        )
    corr1_key = [
        cv2.KeyPoint(corr1[i, 0], corr1[i, 1], radius1) for i in range(corr1.shape[0])
    ]
    corr2_key = [
        cv2.KeyPoint(corr2[i, 0], corr2[i, 1], radius2) for i in range(corr2.shape[0])
    ]

    assert len(corr1) == len(corr2)

    draw_matches = [cv2.DMatch(i, i, 0) for i in range(len(corr1))]
    if color is None:
        color = [(0, 255, 0) if cur_inlier else (0, 0, 255) for cur_inlier in inlier]
    if len(color) == 1:
        display = cv2.drawMatches(
            img1,
            corr1_key,
            img2,
            corr2_key,
            draw_matches,
            None,
            matchColor=color[0],
            singlePointColor=color[0],
            flags=4,
        )
    else:
        height, width = max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1]
        display = np.zeros([height, width, 3], np.uint8)
        display[: img1.shape[0], : img1.shape[1]] = img1
        display[: img2.shape[0], img1.shape[1] :] = img2
        for i in range(len(corr1)):
            left_x, left_y, right_x, right_y = (
                int(corr1[i][0]),
                int(corr1[i][1]),
                int(corr2[i][0] + img1.shape[1]),
                int(corr2[i][1]),
            )
            cur_color = (int(color[i][0]), int(color[i][1]), int(color[i][2]))
            cv2.line(
                display,
                (left_x, left_y),
                (right_x, right_y),
                cur_color,
                1,
                lineType=cv2.LINE_AA,
            )
    return display
