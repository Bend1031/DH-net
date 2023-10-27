"""superpoint+superglue+magsac
"""
import glob
import logging
from time import perf_counter

import cv2  # type: ignore
import hydra
import numpy as np
from tqdm import tqdm

from components import load_component
from lib.config import Config
from lib.utils import pix2pix_RMSE


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="eval_sp_sg",
)
def main(config: Config):
    log = logging.getLogger(__name__)
    dataset = config.dataset.dataset_path
    # 读取图片路径
    imgfiles1 = glob.glob(dataset + r"/test/opt/*.png")
    imgfiles2 = glob.glob(dataset + r"/test/sar/*.png")

    mRMSE = []
    mNCM = []
    mCMR = []
    success_num = 0

    # %% extractor提取特征点和描述子
    extractor = load_component("extractor", config.extractor.name, config.extractor)
    matcher = load_component("matcher", config.matcher.name, config.matcher)
    ransac = load_component("ransac", config.ransac.name, config.ransac)

    start = perf_counter()
    for i in tqdm(range(len(imgfiles1))):
        img1_path = imgfiles1[i]
        img2_path = imgfiles2[i]

        # (height, width, channels)
        img1, img2 = cv2.imread(img1_path), cv2.imread(img2_path)

        # (width, height)
        size1, size2 = np.flip(np.asarray(img1.shape[:2])), np.flip(
            np.asarray(img2.shape[:2])
        )
        # (num_keypoints, 3):(x,y,score) , (num_keypoints, 128)
        # start = time.perf_counter()
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
        # 匹配点的坐标 (num_keypoints, 2), (num_keypoints, 2)
        corr1, corr2 = matcher.run(test_data)

        # %% ransac
        if corr1.shape[0] < 4 or corr2.shape[0] < 4:
            # print("匹配后离散点去除后点数过少")
            continue
        else:
            H, corr1, corr2 = ransac.run(corr1, corr2)
            if corr1.shape[0] < 4 or corr2.shape[0] < 4:
                # print("ransac后离散点去除后点数过少")
                continue

            RMSE, NCM, CMR, bool_list = pix2pix_RMSE(corr1, corr2)
            if NCM >= 5:
                success_num += 1
            mNCM.append(NCM)
            mCMR.append(CMR)
            mRMSE.append(RMSE)
    # 去除0值

    mNCM = [x for x in mNCM if x != 0]
    mCMR = [x for x in mCMR if x != 0]
    mRMSE = [x for x in mRMSE if x != 0]
    end = perf_counter()
    # 数据分析
    # 记录数据集
    log.info(f"数据集为{dataset}")
    log.info(f"NCM均值为{np.mean(mNCM)}")
    log.info(f"CMR均值为{np.mean(mCMR)}")
    log.info(f"RMSE均值为{np.mean(mRMSE)}")
    log.info(f"总数据量为{len(imgfiles1)}")
    log.info(f"成功率为{success_num / len(imgfiles1)}")
    log.info(f"匹配成功数为{success_num}")
    log.info(f"耗时{end - start}s")


if __name__ == "__main__":
    main()
