"""统一所有方法,对一对图片进行配准测试

Usage:
    $ python main_multi_sopatch.py -m
log-dir:
    log/multirun
"""

import glob
import json
import logging
import time

import cv2  # type:ignore
import hydra
import numpy as np
from tqdm import tqdm

from components import load_component

# from lib.config import Config
from lib.rootpath import rootPath
from lib.utils import pix2pix_RMSE


@hydra.main(
    version_base=None,
    config_path="conf",
    # config_name="config_multi_method.yaml",
    config_name="test",
)
def main(config):
    log = logging.getLogger(__name__)
    # 读取图片路径

    method = config.method

    dataset = config.dataset.dataset_path
    imgfiles1 = glob.glob(str(rootPath / dataset) + r"/test/opt/*.png")
    imgfiles2 = glob.glob(str(rootPath / dataset) + r"/test/sar/*.png")

    # %% load component
    extractor = load_component("extractor", method.extractor.name, method.extractor)
    matcher = load_component("matcher", method.matcher.name, method.matcher)
    ransac = load_component("ransac", method.ransac.name, method.ransac)

    mRMSE = []
    mNCM = []
    mCMR = []
    mERR = {}
    success_num = 0
    h_failed = 0
    dists_homo = []
    start_time = time.perf_counter()

    for i in tqdm(range(len(imgfiles1))):
        img1_path = imgfiles1[i]
        img2_path = imgfiles2[i]

        img1, img2 = cv2.imread(img1_path), cv2.imread(img2_path)

        # (width, height)
        size1, size2 = np.flip(np.asarray(img1.shape[:2])), np.flip(
            np.asarray(img2.shape[:2])
        )
        # %% extractor提取特征点和描述子
        kpt1, desc1 = extractor.run(img1_path)
        kpt2, desc2 = extractor.run(img2_path)

        # %% matcher
        test_data = {
            "x1": kpt1,
            "x2": kpt2,
            "desc1": desc1,
            "desc2": desc2,
            "size1": size1,
            "size2": size2,
        }
        corr1, corr2 = matcher.run(test_data)
        if len(corr1) <= 4 or len(corr2) <= 4:
            continue

        # %% ransac
        H_pred, corr1, corr2 = ransac.run(corr1, corr2)
        if len(corr1) <= 4 or len(corr2) <= 4:
            continue
        # %%evaluation homography estimation

        if H_pred is None:
            corner_dist = np.nan
            irat = 0
            h_failed += 1
            inliers = []
        else:
            h, w = img1.shape[:2]
            corners = np.array(
                [[0, 0, 1], [0, h - 1, 1], [w - 1, 0, 1], [w - 1, h - 1, 1]]
            )
            # h_gt 为单位阵
            H_gt = np.eye(3)
            real_warped_corners = np.dot(corners, np.transpose(H_gt))
            real_warped_corners = (
                real_warped_corners[:, :2] / real_warped_corners[:, 2:]
            )
            warped_corners = np.dot(corners, np.transpose(H_pred))
            warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
            corner_dist = np.mean(
                np.linalg.norm(real_warped_corners - warped_corners, axis=1)
            )
            dists_homo.append(corner_dist)
        # %%evaluation

        RMSE, NCM, CMR, bool_list, err = pix2pix_RMSE(corr1, corr2)
        if NCM >= 5:
            success_num += 1
            mNCM.append(NCM)
            mCMR.append(CMR)
            mRMSE.append(RMSE)
            for key in err:
                mERR[key] = mERR.get(key, 0) + err[key]

    thres = [1, 3, 5, 10]
    homo_acc = np.mean(
        [[float(dist <= t) for t in thres] for dist in dists_homo], axis=0
    )
    homo_acc = {t: acc for t, acc in zip(thres, homo_acc)}
    print(homo_acc)

    for key in mERR:
        mERR[key] /= success_num
    # save err
    err = {
        "method": f"{method.name}",
        "dataset": f"{config.dataset.name}",
        "err": mERR,
        "homo_acc": homo_acc,
    }
    with open(
        f"result/{method.name}_{config.dataset.name}.json",
        "w",
    ) as f:
        json.dump(err, f)

    end_time = time.perf_counter()

    # %% 数据分析
    log.info(f"Method:{method.name}")
    log.info(f"Dataset:{config.dataset.name}")
    log.info(f"Dataset size:{len(imgfiles1)}")
    log.info(f"Success num:{success_num}")
    log.info(f"Success rate:{success_num / len(imgfiles1):.3f}")
    log.info(f"NCM:{np.mean(mNCM):.1f}")
    log.info(f"CMR:{np.mean(mCMR):.2f}")
    log.info(f"RMSE:{np.mean(mRMSE):.2f}")
    log.info(f"Time:{int(end_time - start_time):}s")
    log.info("\n")


if __name__ == "__main__":
    main()
