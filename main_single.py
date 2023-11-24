"""统一所有方法,对一对图片进行配准测试
"""

import json
import logging
import time

import cv2  # type:ignore
import hydra
import numpy as np
from matplotlib import pyplot as plt

from components import load_component
from lib.config import Config
from lib.eval_match import img_align
from lib.rootpath import rootPath
from lib.utils import pix2pix_RMSE
from utils import evaluation_utils


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="eval_d2_bf",
)
def main(config: Config):
    log = logging.getLogger(__name__)

    # 读取图片路径
    # path1 = r"D:\Code\LoFTR\20231122162112.jpg"
    # path2 = r"D:\Code\LoFTR\20231122162117.jpg"
    path1 = "datasets/SOPatch/OSdataset/test/opt/d20001.png"
    path2 = "datasets/SOPatch/OSdataset/test/sar/d20001.png"
    log.info(f"img_pair:{path1, path2}")

    img1_path = str(rootPath / path1)
    img2_path = str(rootPath / path2)

    # %% extractor提取特征点和描述子
    extractor = load_component("extractor", config.extractor.name, config.extractor)

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # (width, height)
    size1, size2 = np.flip(np.asarray(img1.shape[:2])), np.flip(
        np.asarray(img2.shape[:2])
    )

    # (num_keypoints, 3):(x,y,score) , (num_keypoints, 128)
    start = time.perf_counter()
    kpt1, desc1 = extractor.run(img1_path)
    kpt2, desc2 = extractor.run(img2_path)
    end = time.perf_counter()
    log.info(
        f"{config.extractor.name}: extract {len(kpt1)},{len(kpt2)} points, extract time: {end - start:.2f}s"
    )

    # %% matcher
    matcher = load_component("matcher", config.matcher.name, config.matcher)
    test_data = {
        "x1": kpt1,
        "x2": kpt2,
        "desc1": desc1,
        "desc2": desc2,
        "size1": size1,
        "size2": size2,
    }
    # 匹配点的坐标 (num_keypoints, 2), (num_keypoints, 2)
    # 匹配耗时
    match_start = time.perf_counter()
    corr1, corr2 = matcher.run(test_data)
    match_end = time.perf_counter()
    log.info(
        f"{config.matcher.name}: match {len(corr1)} points,match time: {match_end - match_start:.2f}s"
    )

    # %% ransac
    ransac = load_component("ransac", config.ransac.name, config.ransac)
    ransac_start = time.perf_counter()
    H, corr1, corr2 = ransac.run(corr1, corr2)
    ransac_end = time.perf_counter()
    log.info(
        f"{config.ransac.name}: match {len(corr1)} points,time: {ransac_end - ransac_start:.4f}s"
    )

    RMSE, NCM, CMR, bool_list, err = pix2pix_RMSE(corr1, corr2)

    # 将err保存为json文件，储存方法名、数据集以及err
    # err = {
    #     "method": f"{config.extractor.name}_{config.matcher.name}_{config.ransac.name}",
    #     "dataset": f"{config.dataset.name}",
    #     "err": err,
    # }
    # 将字典保存为JSON文件

    # with open(
    #     f"{config.extractor.name}_{config.matcher.name}_{config.ransac.name}_{config.dataset.name}.json",
    #     "w",
    # ) as f:
    #     json.dump(err, f)

    log.info(f"NCM:{np.mean(NCM):.2f}")
    log.info(f"CMR:{np.mean(CMR):.2f}")
    log.info(f"RMSE:{np.mean(RMSE):.2f}")

    # %% evaluation
    # show align image
    # img_align(img1, img2, H)

    # visualize match
    # display = evaluation_utils.draw_match(
    #     img1,
    #     img2,
    #     corr1,
    #     corr2,
    #     inlier=bool_list,
    #     radius1=3,
    #     radius2=3,
    # )

    # cv2.imshow("match", display)
    # cv2.waitKey(0)


# cv2.imwrite(
#     f"{config.extractor.name}_{config.matcher.name}_{config.ransac.name}.png",
#     display,
# )

# log.info("match result saved in match.png")
# 对提取出的corr1利用H矩阵进行变换
# corr1转换为齐次坐标
# corr1_homogeneous = np.hstack((corr1, np.ones((len(corr1), 1))))
# # corr1_homogeneous = np.hstack((corr1, np.ones((, 1))))

# # 使用变换矩阵H对点进行变换
# transformed_points = np.dot(H, corr1_homogeneous.T).T

# # 如果需要，将变换后的点转回二维坐标
# transformed_points_2d = transformed_points[:, :2] / transformed_points[:, 2:]
# RMSE, NCM, CMR, bool_list = pix2pix_RMSE(transformed_points_2d, corr2)
# print(f"NCM:{np.mean(NCM):.2f}")
# print(f"CMR:{np.mean(CMR):.2f}")
# print(f"RMSE:{np.mean(RMSE):.2f}")


if __name__ == "__main__":
    main()
