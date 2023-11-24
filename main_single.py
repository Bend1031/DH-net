"""统一所有方法,对一对图片进行配准测试
"""

import json
import logging
import time

import cv2  # type:ignore
import hydra
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

from components import load_component

# from lib.config import Config
from lib.eval_match import img_align
from lib.rootpath import rootPath
from lib.utils import pix2pix_RMSE
from utils import evaluation_utils


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config_single",
)
def main(config):
    print(OmegaConf.to_yaml(config))
    log = logging.getLogger(__name__)

    method = config.method
    # 读取图片路径
    path1 = "datasets/SOPatch/OSdataset/test/opt/d20003.png"
    path2 = "datasets/SOPatch/OSdataset/test/sar/d20003.png"

    log.info(f"img_pair:{path1, path2}")

    img1_path = str(rootPath / path1)
    img2_path = str(rootPath / path2)

    # %% extractor提取特征点和描述子
    extractor = load_component("extractor", method.extractor.name, method.extractor)

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
        f"{method.extractor.name}: extract {len(kpt1)},{len(kpt2)} points, extract time: {end - start:.2f}s"
    )

    # %% matcher
    matcher = load_component("matcher", method.matcher.name, method.matcher)
    kpts_data = {
        "x1": kpt1,
        "x2": kpt2,
        "desc1": desc1,
        "desc2": desc2,
        "size1": size1,
        "size2": size2,
    }
    # 匹配点的坐标 (num_keypoints, 2), (num_keypoints, 2)
    match_start = time.perf_counter()
    corr1, corr2 = matcher.run(kpts_data)
    match_end = time.perf_counter()
    log.info(
        f"{method.matcher.name}: match {len(corr1)} points,match time: {match_end - match_start:.2f}s"
    )

    # %% ransac
    ransac = load_component("ransac", method.ransac.name, method.ransac)
    ransac_start = time.perf_counter()
    H, corr1, corr2 = ransac.run(corr1, corr2)
    ransac_end = time.perf_counter()
    log.info(
        f"{method.ransac.name}: match {len(corr1)} points,time: {ransac_end - ransac_start:.4f}s"
    )

    RMSE, NCM, CMR, bool_list, err = pix2pix_RMSE(corr1, corr2)

    mERR = {}
    for key in err:
        mERR[key] = mERR.get(key, 0) + round(err[key], 3)

    print(mERR)
    # %% 数据分析
    log.info(f"Method:{method.name}")
    log.info(f"NCM:{np.mean(NCM):.1f}")
    log.info(f"CMR:{np.mean(CMR):.2f}")
    log.info(f"RMSE:{np.mean(RMSE):.2f}")

    # %% evaluation
    # show align image
    # img_align(img1, img2, H)

    # # visualize match
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
    #     f"{method.name}.png",
    #     display,
    # )


if __name__ == "__main__":
    main()
