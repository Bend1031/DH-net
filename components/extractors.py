import os
import sys

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from kornia.io import ImageLoadType
from PIL import Image

from lib.model_test import D2Net
from lib.pyramid import process_multiscale
from lib.utils import preprocess_image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from lib.rootpath import rootPath
from superpoint import SuperPoint


def resize(img, resize):
    """ """
    img_h, img_w = img.shape[0], img.shape[1]
    cur_size = max(img_h, img_w)
    if len(resize) == 1:
        scale1, scale2 = resize[0] / cur_size, resize[0] / cur_size
    else:
        scale1, scale2 = resize[0] / img_h, resize[1] / img_w
    new_h, new_w = int(img_h * scale1), int(img_w * scale2)
    new_img = cv2.resize(img.astype("float32"), (new_w, new_h)).astype("uint8")
    scale = np.asarray([scale2, scale1])
    return new_img, scale


def resize_image_with_pil(image, scale_factor):
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    input_pil_image = Image.fromarray(image.astype("uint8"))
    resized_image = input_pil_image.resize((new_width, new_height), Image.ANTIALIAS)
    return np.array(resized_image).astype("float")


class ExtractSIFT:
    def __init__(self, config, root=True):
        self.num_kp = config["num_kpt"]
        self.contrastThreshold = config["det_th"]
        self.resize = config["resize"]
        self.root = root

    def run(self, img_path):
        self.sift = cv2.SIFT_create(
            nfeatures=self.num_kp, contrastThreshold=self.contrastThreshold
        )
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        scale = [1, 1]
        if self.resize[0] != -1:
            img, scale = resize(img, self.resize)
        cv_kp, desc = self.sift.detectAndCompute(img, None)
        kp = np.array(
            [
                [_kp.pt[0] / scale[1], _kp.pt[1] / scale[0], _kp.response]
                for _kp in cv_kp
            ]
        )  # N*3
        index = np.flip(np.argsort(kp[:, 2]))
        kp, desc = kp[index], desc[index]
        if self.root:
            desc = np.sqrt(
                abs(desc / (np.linalg.norm(desc, axis=-1, ord=1)[:, np.newaxis] + 1e-8))
            )
        return kp[: self.num_kp], desc[: self.num_kp]


class ExtractSuperpoint(object):
    def __init__(self, config):
        default_config = {
            "descriptor_dim": 256,
            "nms_radius": 4,
            "detection_threshold": config["det_th"],
            "max_keypoints": config["num_kpt"],
            "remove_borders": 4,
            "model_path": rootPath / "weights/sp/superpoint_v1.pth",
        }
        self.superpoint_extractor = SuperPoint(default_config)
        self.superpoint_extractor.eval(), self.superpoint_extractor.cuda()
        self.num_kp = config["num_kpt"]
        if "padding" in config.keys():
            self.padding = config["padding"]
        else:
            self.padding = False
        self.resize = config["resize"]

    def run(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        scale = 1
        if self.resize[0] != -1:
            img, scale = resize(img, self.resize)
        with torch.inference_mode():
            result = self.superpoint_extractor(
                torch.from_numpy(img / 255.0).float()[None, None].cuda()
            )
        score, kpt, desc = (
            result["scores"][0],
            result["keypoints"][0],
            result["descriptors"][0],
        )
        score, kpt, desc = score.cpu().numpy(), kpt.cpu().numpy(), desc.cpu().numpy().T
        kpt = np.concatenate([kpt / scale, score[:, np.newaxis]], axis=-1)
        # padding randomly
        if self.padding:
            if len(kpt) < self.num_kp:
                res = int(self.num_kp - len(kpt))
                pad_x, pad_desc = np.random.uniform(size=[res, 2]) * (
                    img.shape[0] + img.shape[1]
                ) / 2, np.random.uniform(size=[res, 256])
                pad_kpt, pad_desc = (
                    np.concatenate([pad_x, np.zeros([res, 1])], axis=-1),
                    pad_desc / np.linalg.norm(pad_desc, axis=-1)[:, np.newaxis],
                )
                kpt, desc = np.concatenate([kpt, pad_kpt], axis=0), np.concatenate(
                    [desc, pad_desc], axis=0
                )
        return kpt, desc


class ExtractD2Net:
    def __init__(self, config):
        use_cuda = torch.cuda.is_available()
        self.model_path = config.model_path
        # Creating CNN model
        self.model = D2Net(model_file=rootPath / self.model_path, use_cuda=use_cuda)
        # model = D2Net(model_file="checkpoints/qxslab/qxs.18.pth", use_cuda=use_cuda)
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        self.multiscale = False
        self.max_edge = 2500
        self.max_sum_edges = 5000
        self.is_split = config.is_split

    def run(self, img_path, scales=[0.25, 0.50, 1.0], nfeatures=-1):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # repeat single channel image to 3 channel
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        if self.is_split:
            h, w = image.shape[:2]
            images = [
                image[: h // 2, : w // 2],
                image[: h // 2, w // 2 :],
                image[h // 2 :, : w // 2],
                image[h // 2 :, w // 2 :],
            ]
        else:
            images = [image]

        keypoints_list = []
        descriptors_list = []
        for idx, image in enumerate(images):
            # Resize image to maximum size.
            resized_image = image

            # 如果最大边大于self.max_edge，则调整大小
            if max(resized_image.shape) > self.max_edge:
                scale_factor = self.max_edge / max(resized_image.shape)
                resized_image = resize_image_with_pil(resized_image, scale_factor)

            # 如果尺寸之和大于self.max_sum_edges，则调整大小
            if sum(resized_image.shape[:2]) > self.max_sum_edges:
                scale_factor = self.max_sum_edges / sum(resized_image.shape[:2])
                resized_image = resize_image_with_pil(resized_image, scale_factor)

            # resize proportion
            fact_i = image.shape[0] / resized_image.shape[0]
            fact_j = image.shape[1] / resized_image.shape[1]

            input_image = preprocess_image(resized_image, preprocessing="torch")
            with torch.inference_mode():
                # Process image with D2-Net
                if self.multiscale:
                    keypoints, scores, descriptors = process_multiscale(
                        torch.tensor(
                            input_image[np.newaxis, :, :, :].astype(np.float32),
                            device=self.device,
                        ),
                        self.model,
                        scales,
                    )
                else:
                    keypoints, scores, descriptors = process_multiscale(
                        torch.tensor(
                            input_image[np.newaxis, :, :, :].astype(np.float32),
                            device=self.device,
                        ),
                        self.model,
                        scales=[1],
                    )

            # Input image coordinates
            keypoints[:, 0] *= fact_i
            keypoints[:, 1] *= fact_j
            # i, j -> u, v
            keypoints = keypoints[:, [1, 0, 2]]

            if nfeatures != -1:
                # 根据scores排序
                scores2 = np.array([scores]).T
                res = np.hstack((scores2, keypoints))
                res = res[np.lexsort(-res[:, ::-1].T)]

                res = np.hstack((res, descriptors))
                # 取前几个
                scores = res[0:nfeatures, 0].copy()
                keypoints = res[0:nfeatures, 1:3].copy()
                descriptors = res[0:nfeatures, 4:].copy()
                del res

            # keypoints+scores
            # keypoints = np.concatenate(
            #     (keypoints[:, [0, 1]], np.array([scores]).T), axis=1
            # )

            # 如果是右半部分或下半部分的图像，对特征点进行平移
            if self.is_split:
                if idx == 1 or idx == 3:
                    keypoints[:, 1] += w // 2
                if idx == 2 or idx == 3:
                    keypoints[:, 0] += h // 2

            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)

        # 在返回之前，将keypoints_list和descriptors_list转换为N*2的格式
        keypoints_array = np.vstack(keypoints_list)
        descriptors_array = np.vstack(descriptors_list)

        return keypoints_array, descriptors_array


class ExtractLoFTR:
    def __init__(self, config):
        self.device = K.utils.get_cuda_device_if_available()
        self.pretrained = config.model
        self.matcher = KF.LoFTR(pretrained=self.pretrained).eval().cuda()
        # pass

    def load_torch_image(self, fname):
        img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.0
        img = K.color.bgr_to_rgb(img)
        return img

    def run(self, img1_path, img2_path):
        # img1 = self.load_torch_image(img1_path)
        # img2 = self.load_torch_image(img2_path)

        # matcher = KF.LoFTR(pretrained=self.pretrained).eval().cuda()
        # # matcher = KF.LoFTR(pretrained="outdoor")

        # input_dict = {
        #     "image0": K.color.rgb_to_grayscale(img1).cuda(),  # LofTR 只在灰度图上作用
        #     "image1": K.color.rgb_to_grayscale(img2).cuda(),
        # }
        # tensor 1*1*H*W

        img1 = K.io.load_image(
            img1_path, ImageLoadType.GRAY32, device=self.device
        ).unsqueeze(0)
        img2 = K.io.load_image(
            img2_path, ImageLoadType.GRAY32, device=self.device
        ).unsqueeze(0)

        input_dict = {
            "image0": img1,  # LofTR 只在灰度图上作用
            "image1": img2,
        }
        # 推理模式
        with torch.inference_mode():
            correspondences = self.matcher(input_dict)
        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        return mkpts0, mkpts1


class ExtractDISK:
    def __init__(self, config):
        self.device = K.utils.get_cuda_device_if_available()
        self.num_features = config.num_features
        self.disk = K.feature.DISK.from_pretrained(config.checkpoint).to(self.device)

    def run(self, img_path):
        img = K.io.load_image(img_path, K.io.ImageLoadType.RGB32, device=self.device)[
            None, ...
        ]

        # 推理模式
        with torch.inference_mode():
            [features] = self.disk(img, self.num_features, pad_if_not_divisible=True)

        # 直接在返回语句中返回关键点和描述符
        return features.keypoints, features.descriptors


class ExtractKeyAff:
    def __init__(self, config):
        self.device = K.utils.get_cuda_device_if_available()
        self.num_features = config.num_features
        self.keyaff = K.feature.KeyNetAffNetHardNet(
            num_features=self.num_features, device=self.device
        ).eval()

    def run(self, img_path):
        img = K.io.load_image(img_path, K.io.ImageLoadType.GRAY32, device=self.device)[
            None, ...
        ]

        # 推理模式
        with torch.inference_mode():
            lafs, _, descriptor = self.keyaff.forward(img)

        # 返回numpy
        # laf = laf.cpu().numpy()
        # kpts = laf[:, :, 2][:, [1, 0]]
        # descriptor = np.squeeze(descriptor.cpu().numpy(), axis=0)
        descriptor = torch.squeeze(descriptor, dim=0)
        return lafs, descriptor
