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
from lib.utils import corr_MMA, pix2pix_RMSE
from utils.evaluation_utils import estimate_homo


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config_multi_method.yaml",
    # config_name="test",
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

    # %% 评价指标
    # 平均提取点数
    mean_extract_nums = []
    # 平均匹配点数
    mean_match_nums = []
    # 去除离散点后的匹配点数
    mean_ransac_nums = []
    # 匹配后mma
    m_corr_mma = {}
    m_ransac_mma = {}
    # 利用角点进行单应性矩阵精度的评估，方式同MMA
    # m_dists_homo_mma = {}
    dists_homo = []
    # 失败情况
    failed_nums = 0

    # 阈值3情况下的指标
    mRMSE = []
    mNCM = []
    mCMR = []
    success_num = 0

    # %%
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

        if len(kpt1) == 0 or len(kpt2) == 0:
            failed_nums += 1
            continue

        mean_extract_nums.append(np.mean([len(kpt1), len(kpt2)], dtype=int))
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
            failed_nums += 1
            continue
        mean_match_nums.append(np.mean([len(corr1), len(corr2)], dtype=int))
        corr_mma = corr_MMA(corr1, corr2)
        for key in corr_mma:
            m_corr_mma[key] = m_corr_mma.get(key, 0) + corr_mma[key]
        # %% ransac
        H_pred, corr1, corr2 = ransac.run(corr1, corr2)
        if len(corr1) <= 4 or len(corr2) <= 4 or H_pred is None:
            failed_nums += 1
            continue
        mean_ransac_nums.append(np.mean([len(corr1), len(corr2)], dtype=int))
        # %%evaluation homography estimation:list
        corner_dist = estimate_homo(img1, H_pred)
        dists_homo.append(corner_dist)
        # %%evaluation
        RMSE, NCM, CMR, bool_list, ransac_mma = pix2pix_RMSE(corr1, corr2)
        if NCM >= 5:
            success_num += 1
            mNCM.append(NCM)
            mCMR.append(CMR)
            mRMSE.append(RMSE)
            for key in ransac_mma:
                m_ransac_mma[key] = m_ransac_mma.get(key, 0) + ransac_mma[key]
        else:
            failed_nums += 1

    # %%数据集精度评估
    mean_extract_nums = int(np.mean(mean_extract_nums))
    # 平均匹配点数
    mean_match_nums = int(np.mean(mean_match_nums))
    # 去除离散点后的匹配点数
    mean_ransac_nums = int(np.mean(mean_ransac_nums))
    # homo
    thresholds = list(range(1, 11))
    homo_acc = np.mean(
        [[float(dist <= t) for t in thresholds] for dist in dists_homo], axis=0
    )
    m_homo_mma = {t: round(acc, 3) for t, acc in zip(thresholds, homo_acc)}
    # print(homo_acc)

    for key in m_corr_mma:
        m_corr_mma[key] = round(m_corr_mma[key] / success_num, 3)
    for key in m_ransac_mma:
        m_ransac_mma[key] = round(m_ransac_mma[key] / success_num, 3)

    end_time = time.perf_counter()

    # %% 数据分析，保存为 json
    # save data
    data = {
        "method": f"{method.name}",
        "dataset": f"{config.dataset.name}",
        "dataset_size": len(imgfiles1),
        "success_num": success_num,
        "failed_nums": failed_nums,
        "success_rate": round(success_num / len(imgfiles1), 3),
        "mean_extract_nums": mean_extract_nums,
        "mean_match_nums": mean_match_nums,
        "mean_ransac_nums": mean_ransac_nums,
        "corr_mma": m_corr_mma,
        "ransac_mma": m_ransac_mma,
        "homo_mma": m_homo_mma,
        "NCM": f"{np.mean(mNCM):.1f}",
        "CMR": f"{np.mean(mCMR):.2f}",
        "RMSE": f"{np.mean(mRMSE):.2f}",
        "Time": f"{int(end_time - start_time):}s",
    }
    with open(
        f"result/{method.name}_{config.dataset.name}.json",
        "w",
    ) as f:
        json.dump(data, f)

    log.info(f"Method:{method.name}")
    log.info(f"Dataset:{config.dataset.name}")
    log.info(f"Dataset size:{len(imgfiles1)}")
    log.info(f"Success num:{success_num}")
    log.info(f"failed_nums: {failed_nums}")
    log.info(f"Success rate:{success_num / len(imgfiles1):.3f}")
    log.info(f"mean_extract_nums: {mean_extract_nums}")
    log.info(f"mean_match_nums: {mean_match_nums}")
    log.info(f"mean_ransac_nums: {mean_ransac_nums}")
    log.info(f"corr_mma: {m_corr_mma}")
    log.info(f"ransac_mma: {m_ransac_mma}")
    log.info(f"homo_mma: {m_homo_mma}")
    log.info(f"NCM:{np.mean(mNCM):.1f}")
    log.info(f"CMR:{np.mean(mCMR):.2f}")
    log.info(f"RMSE:{np.mean(mRMSE):.2f}")
    log.info(f"Time:{int(end_time - start_time):}s")
    log.info("\n")


if __name__ == "__main__":
    main()

# %%
