import os
import sys
from collections import OrderedDict, namedtuple

import cv2  # type:ignore
import kornia as K
import kornia.feature as KF
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
from lib.rootpath import rootPath
from sgmnet import matcher as SGM_Model
from superglue import matcher as SG_Model
from utils import evaluation_utils


class GNN_Matcher(object):
    def __init__(self, config, model_name):
        assert model_name == "SGM" or model_name == "SG"

        config = namedtuple("config", config.keys())(*config.values())
        self.p_th = config.p_th
        self.model = SGM_Model(config) if model_name == "SGM" else SG_Model(config)
        self.model.cuda()
        self.model.eval()
        checkpoint = torch.load(
            os.path.join(str(rootPath / config.model_dir), "model_best.pth")
        )
        # for ddp model
        if list(checkpoint["state_dict"].items())[0][0].split(".")[0] == "module":
            new_stat_dict = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                new_stat_dict[key[7:]] = value
            checkpoint["state_dict"] = new_stat_dict
        self.model.load_state_dict(checkpoint["state_dict"])

    def run(self, test_data):
        norm_x1, norm_x2 = evaluation_utils.normalize_size(
            test_data["x1"][:, :2], test_data["size1"]
        ), evaluation_utils.normalize_size(test_data["x2"][:, :2], test_data["size2"])

        x1, x2 = np.concatenate(
            [norm_x1, test_data["x1"][:, 2, np.newaxis]], axis=-1
        ), np.concatenate([norm_x2, test_data["x2"][:, 2, np.newaxis]], axis=-1)

        feed_data = {
            "x1": torch.from_numpy(x1[np.newaxis]).cuda().float(),
            "x2": torch.from_numpy(x2[np.newaxis]).cuda().float(),
            "desc1": torch.from_numpy(test_data["desc1"][np.newaxis]).cuda().float(),
            "desc2": torch.from_numpy(test_data["desc2"][np.newaxis]).cuda().float(),
        }
        try:
            with torch.inference_mode():
                res = self.model(feed_data, test_mode=True)
                p = res["p"]
        except RuntimeError:
            corr1, corr2 = np.array([]), np.array([])
        else:
            index1, index2 = self.match_p(p[0, :-1, :-1])
            corr1, corr2 = (
                test_data["x1"][:, :2][index1.cpu()],
                test_data["x2"][:, :2][index2.cpu()],
            )
            if len(corr1.shape) == 1:
                corr1, corr2 = corr1[np.newaxis], corr2[np.newaxis]
        return corr1, corr2

    def match_p(self, p):  # p N*M
        score, index = torch.topk(p, k=1, dim=-1)
        _, index2 = torch.topk(p, k=1, dim=-2)
        mask_th, index, index2 = score[:, 0] > self.p_th, index[:, 0], index2.squeeze(0)
        mask_mc = index2[index] == torch.arange(len(p)).cuda()
        mask = mask_th & mask_mc
        index1, index2 = torch.nonzero(mask).squeeze(1), index[mask]
        return index1, index2


class NN_Matcher(object):
    def __init__(self, config):
        # config = namedtuple("config", config.keys())(*config.values())
        self.mutual_check = config.mutual_check

    def run(self, test_data):
        desc1, desc2, x1, x2 = (
            torch.from_numpy(test_data["desc1"]),
            torch.from_numpy(test_data["desc2"]),
            torch.from_numpy(test_data["x1"]),
            torch.from_numpy(test_data["x2"]),
        )

        # 使用kornia的match_nn函数计算最近邻匹配
        if self.mutual_check:
            dis, idx = K.feature.match_mnn(desc1, desc2)
        else:
            dis, idx = K.feature.match_nn(desc1, desc2)

        # 计算互相匹配
        corr1, corr2 = x1[:, :2][idx[:, 0]], x2[:, :2][idx[:, 1]]

        return corr1.numpy(), corr2.numpy()


class FLANN_Matcher:
    def __init__(self, config):
        self.ratio_threshold = config.ratio_threshold
        self.is_cmm = config.is_CMM

    def run(self, test_data):
        desc1, desc2, x1, x2 = (
            test_data["desc1"],
            test_data["desc2"],
            test_data["x1"],
            test_data["x2"],
        )

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=40)

        matches = cv2.FlannBasedMatcher(index_params, search_params).knnMatch(
            desc1, desc2, k=2
        )

        if self.is_cmm:
            disdif_avg = sum(n.distance - m.distance for m, n in matches) / len(matches)
            good_matches = [
                m for m, n in matches if n.distance > m.distance + disdif_avg
            ]
        else:
            good_matches = [
                m for m, n in matches if m.distance < self.ratio_threshold * n.distance
            ]

        # 直接在返回语句中创建corr1和corr2，不使用cv2.KeyPoint
        return (
            np.array([x1[m.queryIdx] for m in good_matches]),
            np.array([x2[m.trainIdx] for m in good_matches]),
        )


class BF_Matcher:
    def __init__(self, config):
        """
        Initializes a BF_Matcher object.

        Args:
            config: A configuration object containing parameters for the BFMatcher.

        Returns:
            None
        """
        self.bf = cv2.BFMatcher(crossCheck=config.crossCheck)

    def run(self, test_data):
        """
        Runs the BFMatcher algorithm on the given test data.

        Args:
            test_data: A dictionary containing the following keys:
                - "desc1": The descriptors of the first set of keypoints.
                - "desc2": The descriptors of the second set of keypoints.
                - "x1": The coordinates of the keypoints in the first image.
                - "x2": The coordinates of the keypoints in the second image.

        Returns:
            A tuple containing two numpy arrays:
                - The keypoints in the first image that have matches.
                - The keypoints in the second image that have matches.
        """
        desc1, desc2, kpts1, kpts2 = (
            test_data["desc1"],
            test_data["desc2"],
            test_data["x1"],
            test_data["x2"],
        )

        matches = self.bf.match(desc1, desc2)

        return (
            np.array([kpts1[m.queryIdx] for m in matches]),
            np.array([kpts2[m.trainIdx] for m in matches]),
        )


class LightGlue_Matcher:
    def __init__(self, config):
        self.extract_model = config.extract_model
        self.device = K.utils.get_cuda_device_if_available()
        self.lightglue = K.feature.LightGlueMatcher(self.extract_model).to(self.device)

    def run(self, test_data):
        desc1, desc2, kps1, kps2 = (
            test_data["desc1"],
            test_data["desc2"],
            test_data["x1"],
            test_data["x2"],
        )
        # numpy->tensor,nx3->nx2
        # 判断是否需要转换为tensor
        if not isinstance(kps1, torch.Tensor):
            kps1, kps2 = (
                torch.from_numpy(kps1[:, :2]).to(self.device),
                torch.from_numpy(kps2[:, :2]).to(self.device),
            )
            desc1, desc2 = (
                torch.from_numpy(desc1).to(self.device),
                torch.from_numpy(desc2).to(self.device),
            )
        else:
            kps1, kps2 = kps1[:, :2].to(self.device), kps2[:, :2].to(self.device)
            desc1, desc2 = desc1.to(self.device), desc2.to(self.device)

        lafs1 = KF.laf_from_center_scale_ori(
            kps1[None], torch.ones(1, len(kps1), 1, 1, device=self.device)
        )
        lafs2 = KF.laf_from_center_scale_ori(
            kps2[None], torch.ones(1, len(kps2), 1, 1, device=self.device)
        )

        with torch.inference_mode():
            dists, idxs = self.lightglue(desc1, desc2, lafs1, lafs2)

        # 直接在返回语句中将corr1和corr2从张量转换为NumPy数组
        return (
            kps1[idxs[:, 0]].detach().cpu().numpy(),
            kps2[idxs[:, 1]].detach().cpu().numpy(),
        )


class AdaLAM_Matcher:
    def __init__(self, config):
        self.device = K.utils.get_cuda_device_if_available()

    def run(self, test_data):
        desc1, desc2, laf1, laf2 = (
            test_data["desc1"],
            test_data["desc2"],
            test_data["x1"],
            test_data["x2"],
        )

        def get_matching_keypoints(lafs1, lafs2, idxs):
            mkpts1 = (
                KF.get_laf_center(lafs1).squeeze()[idxs[:, 0]].detach().cpu().numpy()
            )
            mkpts2 = (
                KF.get_laf_center(lafs2).squeeze()[idxs[:, 1]].detach().cpu().numpy()
            )
            return mkpts1, mkpts2

        with torch.inference_mode():
            dists, idxs = KF.match_adalam(desc1, desc2, laf1, laf2)
        corr1, corr2 = get_matching_keypoints(laf1, laf2, idxs)
        return corr1, corr2
