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
        self.threshold = config.threshold
        self.max_iters = config.max_iters
        self.confidence = config.confidence

    def run(self, corr1, corr2):
        H, inliers = cv2.findHomography(
            srcPoints=corr1,
            dstPoints=corr2,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.threshold,
            confidence=self.confidence,
            maxIters=self.max_iters,
        )
        inlier_idxs = np.nonzero(inliers)[0]
        # print(inlier_idxs)
        corr1 = corr1[inlier_idxs]
        corr2 = corr2[inlier_idxs]
        return H, corr1, corr2


class Degensac:
    def __init__(self, config):
        self.threshold = config.threshold
        self.max_iters = config.max_iters
        self.confidence = config.confidence

    def run(self, corr1, corr2):
        H, inliers = pydegensac.findHomography(
            pts1_=corr1,
            pts2_=corr2,
            px_th=self.threshold,
            conf=self.confidence,
            max_iters=self.max_iters,
        )
        inlier_idxs = np.nonzero(inliers)[0]
        # print(inlier_idxs)
        corr1 = corr1[inlier_idxs]
        corr2 = corr2[inlier_idxs]
        return H, corr1, corr2


class Magsacpp:
    def __init__(self, config):
        self.threshold = config.threshold
        self.max_iters = config.max_iters
        self.confidence = config.confidence

    def run(self, corr1, corr2):
        H, inliers = cv2.findHomography(
            srcPoints=corr1,
            dstPoints=corr2,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=self.threshold,
            confidence=self.confidence,
            maxIters=self.max_iters,
        )
        inlier_idxs = np.nonzero(inliers)[0]
        # print(inlier_idxs)
        corr1 = corr1[inlier_idxs]
        corr2 = corr2[inlier_idxs]
        return H, corr1, corr2
