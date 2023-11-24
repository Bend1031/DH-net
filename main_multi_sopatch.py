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
from lib.config import Config
from lib.rootpath import rootPath
from lib.utils import pix2pix_RMSE


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config_multi_method",
)
def main(config_: Config):
    log = logging.getLogger(__name__)
    # 读取图片路径
    config = config_.method
    dataset = config.dataset.dataset_path
    imgfiles1 = glob.glob(str(rootPath / dataset) + r"/test/opt/*.png")
    imgfiles2 = glob.glob(str(rootPath / dataset) + r"/test/sar/*.png")

    log.info(f"Dataset:{dataset}")

    # %% load component
    extractor = load_component("extractor", config.extractor.name, config.extractor)
    matcher = load_component("matcher", config.matcher.name, config.matcher)
    ransac = load_component("ransac", config.ransac.name, config.ransac)

    mRMSE = []
    mNCM = []
    mCMR = []
    mERR = {}
    success_num = 0

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
        H, corr1, corr2 = ransac.run(corr1, corr2)
        if len(corr1) <= 4 or len(corr2) <= 4:
            continue
        # %%evaluation

        RMSE, NCM, CMR, bool_list, err = pix2pix_RMSE(corr1, corr2)
        if NCM >= 5:
            success_num += 1
            mNCM.append(NCM)
            mCMR.append(CMR)
            mRMSE.append(RMSE)
            for key in err:
                mERR[key] = mERR.get(key, 0) + err[key]

    for key in mERR:
        mERR[key] /= success_num
    # save err
    err = {
        "method": f"{config.extractor.name}_{config.matcher.name}_{config.ransac.name}",
        "dataset": f"{config.dataset.name}",
        "err": mERR,
    }
    with open(
        f"result/{config.extractor.name}_{config.matcher.name}_{config.ransac.name}_{config.dataset.name}.json",
        "w",
    ) as f:
        json.dump(err, f)

    end_time = time.perf_counter()

    # %% 数据分析
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
