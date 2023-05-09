import os
import sys

import numpy as np
from skimage import measure, transform

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
import cv2
import pydegensac


class RANSAC:
    def __init__(self, config):
        # self.model_class = config.model_class
        # self.min_samples = config.min_samples
        # self.residual_threshold = config.residual_threshold
        # self.max_trials = config.max_trials
        pass

    def run(self, corr1, corr2):
        Fm, inliers = cv2.findFundamentalMat(
            corr1, corr2, cv2.RANSAC, 3.0, 0.999, 10000
        )
        inlier_idxs = np.nonzero(inliers)[0]
        # print(inlier_idxs)
        corr1 = corr1[inlier_idxs]
        corr2 = corr2[inlier_idxs]
        return corr1, corr2
        # ransac_model, inliers = measure.ransac(
        #     data=(corr1, corr2),
        #     model_class=transform.AffineTransform,
        #     min_samples=4,
        #     residual_threshold=30,
        #     max_trials=1000,
        # )
        # inlier_idxs = np.nonzero(inliers)[0]
        # # print(inlier_idxs)
        # corr1 = corr1[inlier_idxs]
        # corr2 = corr2[inlier_idxs]
        # return corr1, corr2


class Degensac:
    def run(self, corr1, corr2):
        model, inliers = pydegensac.findHomography(corr1, corr2, 4.0, 0.999, 1000)
        inlier_idxs = np.nonzero(inliers)[0]
        # print(inlier_idxs)
        corr1 = corr1[inlier_idxs]
        corr2 = corr2[inlier_idxs]
        return corr1, corr2


class Magsac:
    def run(self, corr1, corr2):
        ransac_result = pydegensac.ransac(
            data=(corr1, corr2),
            model_class=transform.AffineTransform,
            method="magsac",
            max_trials=1000,
            threshold=0.01,
            verbose=False,
        )
        inlier_idxs = ransac_result.inliers
        # inlier_idxs = np.nonzero(inliers)[0]
        # print(inlier_idxs)
        corr1 = corr1[inlier_idxs]
        corr2 = corr2[inlier_idxs]
        return corr1, corr2


class Magsacpp:
    def __init__(self, config):
        pass

    def run(self, corr1, corr2):
        Fm, inliers = cv2.findFundamentalMat(
            corr1, corr2, cv2.USAC_MAGSAC, 1.0, 0.99, 1000
        )
        inlier_idxs = np.nonzero(inliers)[0]
        # print(inlier_idxs)
        corr1 = corr1[inlier_idxs]
        corr2 = corr2[inlier_idxs]
        return corr1, corr2
