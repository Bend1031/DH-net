"""superpoint+superglue+magsac 单图匹配
"""
import time

import cv2
import hydra
import numpy as np

from components import load_component
from lib.config import Config
from lib.rootpath import rootPath
from lib.utils import pix2pix_RMSE
from utils import evaluation_utils


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="eval_sp_sg",
)
def main(config: Config):
    # 读取图片路径
    path1 = "datasets/test_dataset/4SARSets/pair8-1.tif"
    path2 = "datasets/test_dataset/4SARSets/pair8-2.tif"

    img1_path = str(rootPath / path1)
    img2_path = str(rootPath / path2)

    # %% extractor提取特征点和描述子
    extractor = load_component("extractor", config.extractor.name, config.extractor)

    # (height, width, channels)
    img1, img2 = cv2.imread(img1_path), cv2.imread(img2_path)

    # (width, height)
    size1, size2 = np.flip(np.asarray(img1.shape[:2])), np.flip(
        np.asarray(img2.shape[:2])
    )
    # (num_keypoints, 3):(x,y,score) , (num_keypoints, 128)
    start = time.perf_counter()
    kpt1, desc1 = extractor.run(img1_path)
    kpt2, desc2 = extractor.run(img2_path)
    end = time.perf_counter()
    print(f"{config.extractor.name} extract time: {end - start:.2f}s")
    print(f"extract {len(kpt1)} points")
    print(f"extract {len(kpt2)} points")

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
    print(f"match {len(corr1)} points")
    print(f"{config.matcher.name} match time: {match_end - match_start:.2f}s")

    # %% ransac
    ransac = load_component("ransac", config.ransac.name, config.ransac)
    ransac_start = time.perf_counter()
    H, corr1, corr2 = ransac.run(corr1, corr2)
    ransac_end = time.perf_counter()
    print(f"after ransac, match {len(corr1)} points")
    print(f"{config.ransac.name} ransac time: {ransac_end - ransac_start:.4f}s")

    RMSE, NCM, CMR, bool_list = pix2pix_RMSE(corr1, corr2)

    print(f"NCM均值为{np.mean(NCM)}")
    print(f"CMR均值为{np.mean(CMR)}")
    print(f"RMSE均值为{np.mean(RMSE)}")

    # %% evaluation
    # show align image
    # img_align(img1, img2, H)

    # visualize match
    display = evaluation_utils.draw_match(
        img1,
        img2,
        corr1,
        corr2,
        inlier=bool_list,
    )

    cv2.imshow("match", display)
    cv2.waitKey(0)
    # cv2.imwrite(
    #     f"{config.extractor.name}_{config.matcher.name}_{config.ransac.name}.png",
    #     display,
    # )

    # print("match result saved in match.png")


if __name__ == "__main__":
    main()
