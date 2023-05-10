import os
import sys
import time

import cv2
import hydra
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
import torch.nn as nn
from omegaconf import OmegaConf

from components import load_component
from lib.config import Config
from lib.eval_match import img_align
from lib.rootpath import rootPath
from utils import evaluation_utils


@hydra.main(
    version_base=None,
    config_path=os.path.join(ROOT_DIR, "conf"),
    config_name="eval_sp_sgm",
)
def main(config: Config):
    img1_path = str(rootPath / r"test_imgs/02/pair1.jpg")
    img2_path = str(rootPath / r"test_imgs/02/pair2.jpg")

    extractor = load_component("extractor", config.extractor.name, config.extractor)
    # (height, width, channels)
    img1, img2 = cv2.imread(img1_path), cv2.imread(img2_path)
    # (width, height)

    size1, size2 = np.flip(np.asarray(img1.shape[:2])), np.flip(
        np.asarray(img2.shape[:2])
    )
    # (num_keypoints, 3):(x,y,score) , (num_keypoints, 128)
    kpt1, desc1 = extractor.run(img1_path)
    kpt2, desc2 = extractor.run(img2_path)
    print(f"extract {len(kpt1)} points")

    # linear = nn.Linear(512, 256)
    # desc1 = linear(torch.from_numpy(desc1)).detach().numpy()
    # desc2 = linear(torch.from_numpy(desc2)).detach().numpy()

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
    print(f"{config.matcher.name} match time: {match_end - match_start:.2f}")

    # ransac
    ransac = load_component("ransac", config.ransac.name, config.ransac)
    ransac_start = time.perf_counter()
    H, corr1, corr2 = ransac.run(corr1, corr2)
    ransac_end = time.perf_counter()
    print(f"after ransac, match {len(corr1)} points")
    print(f"{config.ransac.name} ransac time: {ransac_end - ransac_start:.4f}")

    # show align image
    img_align(img1, img2, H)

    # draw points
    dis_points_1 = evaluation_utils.draw_points(img1, kpt1)
    dis_points_2 = evaluation_utils.draw_points(img2, kpt2)

    # visualize match
    display = evaluation_utils.draw_match(dis_points_1, dis_points_2, corr1, corr2)
    cv2.imwrite(
        f"{config.extractor.name}_{config.matcher.name}_{config.ransac.name}.png",
        display,
    )
    print("match result saved in match.png")


if __name__ == "__main__":
    main()
