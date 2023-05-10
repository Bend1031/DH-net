import cv2
import numpy as np


def flann(kps_left, des_left, kps_right, des_right):
    """
    # 优点：批量特征匹配时，FLANN速度快；
    # 缺点：由于使用的是邻近近似值，所以精度较差
    # Index_params字典：匹配算法KDTREE,LSH;
    # Search_parames字典:指定KDTREE算法中遍历树的次数；
    """
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
    # print("match num is %d" % len(goodMatch))
    locations_1_to_use = np.array(locations_1_to_use)
    locations_2_to_use = np.array(locations_2_to_use)
    return locations_1_to_use, locations_2_to_use


def img_align(img1, img2, H):
    """
    img_align - Function to align two images using homography matrix H

    Parameters:
    img1 (numpy.ndarray): Input image 1
    img2 (numpy.ndarray): Input image 2
    H (numpy.ndarray): Homography matrix to align img1 with img2

    Returns:
    None

    Algorithm:
    1. Extract height and width of img2
    2. Use cv2.warpPerspective to warp img1 using H to match the perspective of img2
    3. Combine the two images horizontally using cv2.hconcat
    4. Display the aligned images using cv2.imshow and wait for user input

    """

    h, w = img2.shape[:2]
    img1_warped = cv2.warpPerspective(img1, H, (w, h))

    # Display the aligned images
    aligned_image = cv2.hconcat([img1_warped, img2])
    cv2.imshow("Aligned Images", aligned_image)
    cv2.waitKey(0)
