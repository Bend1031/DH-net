import glob
import os
import time

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from lib.utils import preprocess_image


class MegaDepthDataset(Dataset):
    def __init__(
        self,
        scene_list_path="megadepth_utils/train_scenes.txt",
        scene_info_path="/local/dataset/megadepth/scene_info",
        base_path="/local/dataset/megadepth",
        train=True,
        preprocessing=None,
        min_overlap_ratio=0.5,
        max_overlap_ratio=1,
        max_scale_ratio=np.inf,
        pairs_per_scene=100,
        image_size=256,
    ):
        self.scenes = []
        with open(scene_list_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.scenes.append(line.strip("\n"))

        self.scene_info_path = scene_info_path
        self.base_path = base_path

        self.train = train

        self.preprocessing = preprocessing

        self.min_overlap_ratio = min_overlap_ratio
        self.max_overlap_ratio = max_overlap_ratio
        self.max_scale_ratio = max_scale_ratio

        self.pairs_per_scene = pairs_per_scene

        self.image_size = image_size

        self.dataset = []

    def build_dataset(self):
        self.dataset = []
        if not self.train:
            np_random_state = np.random.get_state()
            np.random.seed(42)
            print("Building the validation dataset...")
        else:
            print("Building a new training dataset...")
        for scene in tqdm(self.scenes, total=len(self.scenes)):
            scene_info_path = os.path.join(self.scene_info_path, "%s.npz" % scene)
            if not os.path.exists(scene_info_path):
                continue
            scene_info = np.load(scene_info_path, allow_pickle=True)
            overlap_matrix = scene_info["overlap_matrix"]
            scale_ratio_matrix = scene_info["scale_ratio_matrix"]

            valid = np.logical_and(
                np.logical_and(
                    overlap_matrix >= self.min_overlap_ratio,
                    overlap_matrix <= self.max_overlap_ratio,
                ),
                scale_ratio_matrix <= self.max_scale_ratio,
            )

            pairs = np.vstack(np.where(valid))
            try:
                selected_ids = np.random.choice(pairs.shape[1], self.pairs_per_scene)
            except:
                continue

            image_paths = scene_info["image_paths"]
            depth_paths = scene_info["depth_paths"]
            points3D_id_to_2D = scene_info["points3D_id_to_2D"]
            points3D_id_to_ndepth = scene_info["points3D_id_to_ndepth"]
            intrinsics = scene_info["intrinsics"]
            poses = scene_info["poses"]

            for pair_idx in selected_ids:
                idx1 = pairs[0, pair_idx]
                idx2 = pairs[1, pair_idx]
                matches = np.array(
                    list(
                        points3D_id_to_2D[idx1].keys() & points3D_id_to_2D[idx2].keys()
                    )
                )

                # Scale filtering
                matches_nd1 = np.array(
                    [points3D_id_to_ndepth[idx1][match] for match in matches]
                )
                matches_nd2 = np.array(
                    [points3D_id_to_ndepth[idx2][match] for match in matches]
                )
                scale_ratio = np.maximum(
                    matches_nd1 / matches_nd2, matches_nd2 / matches_nd1
                )
                matches = matches[np.where(scale_ratio <= self.max_scale_ratio)[0]]

                point3D_id = np.random.choice(matches)
                point2D1 = points3D_id_to_2D[idx1][point3D_id]
                point2D2 = points3D_id_to_2D[idx2][point3D_id]
                nd1 = points3D_id_to_ndepth[idx1][point3D_id]
                nd2 = points3D_id_to_ndepth[idx2][point3D_id]
                central_match = np.array(
                    [point2D1[1], point2D1[0], point2D2[1], point2D2[0]]
                )
                self.dataset.append(
                    {
                        "image_path1": image_paths[idx1],
                        "depth_path1": depth_paths[idx1],
                        "intrinsics1": intrinsics[idx1],
                        "pose1": poses[idx1],
                        "image_path2": image_paths[idx2],
                        "depth_path2": depth_paths[idx2],
                        "intrinsics2": intrinsics[idx2],
                        "pose2": poses[idx2],
                        "central_match": central_match,
                        "scale_ratio": max(nd1 / nd2, nd2 / nd1),
                    }
                )
        np.random.shuffle(self.dataset)
        if not self.train:
            np.random.set_state(np_random_state)

    def __len__(self):
        return len(self.dataset)

    def recover_pair(self, pair_metadata):
        depth_path1 = os.path.join(self.base_path, pair_metadata["depth_path1"])
        with h5py.File(depth_path1, "r") as hdf5_file:
            depth1 = np.array(hdf5_file["/depth"])
        assert np.min(depth1) >= 0
        image_path1 = os.path.join(self.base_path, pair_metadata["image_path1"])
        image1 = Image.open(image_path1)
        if image1.mode != "RGB":
            image1 = image1.convert("RGB")
        image1 = np.array(image1)
        assert image1.shape[0] == depth1.shape[0] and image1.shape[1] == depth1.shape[1]
        intrinsics1 = pair_metadata["intrinsics1"]
        pose1 = pair_metadata["pose1"]

        depth_path2 = os.path.join(self.base_path, pair_metadata["depth_path2"])
        with h5py.File(depth_path2, "r") as hdf5_file:
            depth2 = np.array(hdf5_file["/depth"])
        assert np.min(depth2) >= 0
        image_path2 = os.path.join(self.base_path, pair_metadata["image_path2"])
        image2 = Image.open(image_path2)
        if image2.mode != "RGB":
            image2 = image2.convert("RGB")
        image2 = np.array(image2)
        assert image2.shape[0] == depth2.shape[0] and image2.shape[1] == depth2.shape[1]
        intrinsics2 = pair_metadata["intrinsics2"]
        pose2 = pair_metadata["pose2"]

        central_match = pair_metadata["central_match"]
        image1, bbox1, image2, bbox2 = self.crop(image1, image2, central_match)

        depth1 = depth1[
            bbox1[0] : bbox1[0] + self.image_size, bbox1[1] : bbox1[1] + self.image_size
        ]
        depth2 = depth2[
            bbox2[0] : bbox2[0] + self.image_size, bbox2[1] : bbox2[1] + self.image_size
        ]

        return (
            image1,
            depth1,
            intrinsics1,
            pose1,
            bbox1,
            image2,
            depth2,
            intrinsics2,
            pose2,
            bbox2,
        )

    def crop(self, image1, image2, central_match):
        bbox1_i = max(int(central_match[0]) - self.image_size // 2, 0)
        if bbox1_i + self.image_size >= image1.shape[0]:
            bbox1_i = image1.shape[0] - self.image_size
        bbox1_j = max(int(central_match[1]) - self.image_size // 2, 0)
        if bbox1_j + self.image_size >= image1.shape[1]:
            bbox1_j = image1.shape[1] - self.image_size

        bbox2_i = max(int(central_match[2]) - self.image_size // 2, 0)
        if bbox2_i + self.image_size >= image2.shape[0]:
            bbox2_i = image2.shape[0] - self.image_size
        bbox2_j = max(int(central_match[3]) - self.image_size // 2, 0)
        if bbox2_j + self.image_size >= image2.shape[1]:
            bbox2_j = image2.shape[1] - self.image_size

        return (
            image1[
                bbox1_i : bbox1_i + self.image_size, bbox1_j : bbox1_j + self.image_size
            ],
            np.array([bbox1_i, bbox1_j]),
            image2[
                bbox2_i : bbox2_i + self.image_size, bbox2_j : bbox2_j + self.image_size
            ],
            np.array([bbox2_i, bbox2_j]),
        )

    def __getitem__(self, idx):
        (
            image1,
            depth1,
            intrinsics1,
            pose1,
            bbox1,
            image2,
            depth2,
            intrinsics2,
            pose2,
            bbox2,
        ) = self.recover_pair(self.dataset[idx])

        image1 = preprocess_image(image1, preprocessing=self.preprocessing)
        image2 = preprocess_image(image2, preprocessing=self.preprocessing)

        return {
            "image1": torch.from_numpy(image1.astype(np.float32)),
            "depth1": torch.from_numpy(depth1.astype(np.float32)),
            "intrinsics1": torch.from_numpy(intrinsics1.astype(np.float32)),
            "pose1": torch.from_numpy(pose1.astype(np.float32)),
            "bbox1": torch.from_numpy(bbox1.astype(np.float32)),
            "image2": torch.from_numpy(image2.astype(np.float32)),
            "depth2": torch.from_numpy(depth2.astype(np.float32)),
            "intrinsics2": torch.from_numpy(intrinsics2.astype(np.float32)),
            "pose2": torch.from_numpy(pose2.astype(np.float32)),
            "bbox2": torch.from_numpy(bbox2.astype(np.float32)),
        }


class QxslabSarOptDataset(Dataset):
    def __init__(
        self,
        scene_list_path="qxslab_utils/train.txt",
        base_path="datasets/QXSLAB_SAROPT/",
        subfolder_opt="opt_256_oc_0.2/",
        subfolder_sar="sar_256_oc_0.2/",
        image_size=256,
        preprocessing=None,
        train=True,
    ):
        self.scenes = []
        with open(scene_list_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.scenes.append(line.strip("\n") + ".png")

        # self.scene_list_path = scene_list_path
        self.base_path = base_path
        self.image_size = image_size
        self.preprocessing = preprocessing
        self.train = train
        self.subfolder_opt = subfolder_opt
        self.subfolder_sar = subfolder_sar

        self.dataset = []

    def build_dataset(self):
        self.dataset = []

        if not self.train:
            np_random_state = np.random.get_state()
            np.random.seed(42)
            print("Building the validation dataset...")
        else:
            print("Building a new training dataset...")

        for scene in tqdm(self.scenes, total=len(self.scenes)):
            image_path1 = os.path.join(self.base_path, self.subfolder_opt, scene)
            image_path2 = os.path.join(self.base_path, self.subfolder_sar, scene)

            self.dataset.append(
                {
                    "image_path1": image_path1,
                    "image_path2": image_path2,
                }
            )
        np.random.shuffle(self.dataset)
        if not self.train:
            np.random.set_state(np_random_state)

    def recover_pair(self, pair_metadata):
        image1 = Image.open(pair_metadata["image_path1"])
        if image1.mode != "RGB":
            image1 = image1.convert("RGB")
        image1 = np.array(image1)

        image2 = Image.open(pair_metadata["image_path2"])
        if image2.mode != "RGB":
            image2 = image2.convert("RGB")
        image2 = np.array(image2)

        return image1, image2

    def __len__(self):
        return len(self.dataset)

    def init_my_params(self):
        self.depth1 = np.ones((self.image_size, self.image_size))
        # self.intrinsics1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.intrinsics1 = np.identity(3)
        # pose=[4,4]
        # self.pose1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.pose1 = np.identity(4)
        self.bbox1 = np.array([0, 0])

        self.depth2 = self.depth1
        self.intrinsics2 = self.intrinsics1
        self.pose2 = self.pose1
        self.bbox2 = self.bbox1
        return (
            self.depth1,
            self.intrinsics1,
            self.pose1,
            self.bbox1,
            self.depth2,
            self.intrinsics2,
            self.pose2,
            self.bbox2,
        )

    def __getitem__(self, idx):
        (
            image1,
            image2,
        ) = self.recover_pair(self.dataset[idx])

        image1 = preprocess_image(image1, preprocessing=self.preprocessing)
        image2 = preprocess_image(image2, preprocessing=self.preprocessing)

        # return {
        #     "image1": torch.from_numpy(image1.astype(np.float32)),
        #     "image2": torch.from_numpy(image2.astype(np.float32)),
        # }
        (
            depth1,
            intrinsics1,
            pose1,
            bbox1,
            depth2,
            intrinsics2,
            pose2,
            bbox2,
        ) = self.init_my_params()
        return {
            "image1": torch.from_numpy(image1.astype(np.float32)),
            "depth1": torch.from_numpy(depth1.astype(np.float32)),
            "intrinsics1": torch.from_numpy(intrinsics1.astype(np.float32)),
            "pose1": torch.from_numpy(pose1.astype(np.float32)),
            "bbox1": torch.from_numpy(bbox1.astype(np.float32)),
            "image2": torch.from_numpy(image2.astype(np.float32)),
            "depth2": torch.from_numpy(depth2.astype(np.float32)),
            "intrinsics2": torch.from_numpy(intrinsics2.astype(np.float32)),
            "pose2": torch.from_numpy(pose2.astype(np.float32)),
            "bbox2": torch.from_numpy(bbox2.astype(np.float32)),
        }


class WhuDataset(Dataset):
    def __init__(
        self,
        scene_list_path="datasets_utils/whu-opt-sar-utils/train.txt",
        # scene_info_path="/local/dataset/megadepth/scene_info",
        subfolder_opt="opt_png/",
        subfolder_sar="sar_png/",
        base_path="datasets/whu-opt-sar/",
        train=True,
        preprocessing=None,
        pairs_per_scene=100,
        image_size=256,
    ):
        self.scenes = []
        with open(scene_list_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.scenes.append(line.strip("\n") + ".png")
        self.base_path = base_path
        self.train = train
        self.preprocessing = preprocessing
        self.pairs_per_scene = pairs_per_scene
        self.subfolder_opt = subfolder_opt
        self.subfolder_sar = subfolder_sar
        self.image_size = image_size

        self.dataset = []

    def generate_uniform_coordinates(self, height, width, grid_size=(10, 10)):
        # height, width = image.shape[:2]
        x_step = width // grid_size[1]
        y_step = height // grid_size[0]

        coordinates = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                x = j * x_step + x_step // 2
                y = i * y_step + y_step // 2
                coordinates.append((x, y, x, y))

        return np.array(coordinates)

    def build_dataset(self):
        self.dataset = []

        if not self.train:
            np_random_state = np.random.get_state()
            np.random.seed(42)
            print("Building the validation dataset...")
        else:
            print("Building a new training dataset...")

        for scene in tqdm(self.scenes, total=len(self.scenes)):
            image_path1 = os.path.join(self.base_path, self.subfolder_opt, scene)
            image_path2 = os.path.join(self.base_path, self.subfolder_sar, scene)

            image = Image.open(image_path1)
            if image.mode != "RGB":
                image = image.convert("RGB")
                # image = np.array(image)

            # numpy shape

            height, width = image.size

            central_match = self.generate_uniform_coordinates(height, width)
            for pair in central_match:
                self.dataset.append(
                    {
                        "image_path1": image_path1,
                        "image_path2": image_path2,
                        "central_match": pair,
                    }
                )

        np.random.shuffle(self.dataset)
        if not self.train:
            np.random.set_state(np_random_state)

    def __len__(self):
        return len(self.dataset)

    def recover_pair(self, pair_metadata):
        # load image to numpy array
        image1 = Image.open(pair_metadata["image_path1"])
        if image1.mode != "RGB":
            image1 = image1.convert("RGB")
        image1 = np.array(image1)

        image2 = Image.open(pair_metadata["image_path2"])
        if image2.mode != "RGB":
            image2 = image2.convert("RGB")
        image2 = np.array(image2)

        central_match = pair_metadata["central_match"]

        image1, bbox1, image2, bbox2 = self.crop(image1, image2, central_match)

        return image1, bbox1, image2, bbox2

    def init_my_params(self):
        self.depth1 = np.ones((self.image_size, self.image_size))
        # self.intrinsics1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.intrinsics1 = np.identity(3)
        # pose=[4,4]
        # self.pose1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.pose1 = np.identity(4)
        # self.bbox1 = np.array([0, 0])

        self.depth2 = self.depth1
        self.intrinsics2 = self.intrinsics1
        self.pose2 = self.pose1
        # self.bbox2 = self.bbox1
        return (
            self.depth1,
            self.intrinsics1,
            self.pose1,
            # self.bbox1,
            self.depth2,
            self.intrinsics2,
            self.pose2,
            # self.bbox2,
        )

    def crop(self, image1, image2, central_match, overlap_ratio=0.5):
        height, width = image1.shape[:2]
        half_crop_size = self.image_size // 2
        overlap_size = int(half_crop_size * overlap_ratio)

        offset_range = half_crop_size - overlap_size
        i_offset1 = np.random.randint(-offset_range, offset_range + 1)
        j_offset1 = np.random.randint(-offset_range, offset_range + 1)
        i_offset2 = np.random.randint(-offset_range, offset_range + 1)
        j_offset2 = np.random.randint(-offset_range, offset_range + 1)

        center1 = np.array([central_match[0] + i_offset1, central_match[1] + j_offset1])
        center2 = np.array([central_match[2] + i_offset2, central_match[3] + j_offset2])

        # 计算裁剪窗口的左上角坐标，并进行边界检查
        bbox1_i = max(min(center1[0] - half_crop_size, height - self.image_size), 0)
        bbox1_j = max(min(center1[1] - half_crop_size, width - self.image_size), 0)

        bbox2_i = max(min(center2[0] - half_crop_size, height - self.image_size), 0)
        bbox2_j = max(min(center2[1] - half_crop_size, width - self.image_size), 0)

        cropped_image1 = image1[
            bbox1_i : bbox1_i + self.image_size, bbox1_j : bbox1_j + self.image_size
        ]
        cropped_image2 = image2[
            bbox2_i : bbox2_i + self.image_size, bbox2_j : bbox2_j + self.image_size
        ]

        return (
            cropped_image1,
            np.array([bbox1_i, bbox1_j]),
            cropped_image2,
            np.array([bbox2_i, bbox2_j]),
        )

    def __getitem__(self, idx):
        (
            image1,
            bbox1,
            image2,
            bbox2,
        ) = self.recover_pair(self.dataset[idx])

        image1 = preprocess_image(image1, preprocessing=self.preprocessing)
        image2 = preprocess_image(image2, preprocessing=self.preprocessing)

        (
            depth1,
            intrinsics1,
            pose1,
            depth2,
            intrinsics2,
            pose2,
        ) = self.init_my_params()

        return {
            "image1": torch.from_numpy(image1.astype(np.float32)),
            "depth1": torch.from_numpy(depth1.astype(np.float32)),
            "intrinsics1": torch.from_numpy(intrinsics1.astype(np.float32)),
            "pose1": torch.from_numpy(pose1.astype(np.float32)),
            "bbox1": torch.from_numpy(bbox1.astype(np.float32)),
            "image2": torch.from_numpy(image2.astype(np.float32)),
            "depth2": torch.from_numpy(depth2.astype(np.float32)),
            "intrinsics2": torch.from_numpy(intrinsics2.astype(np.float32)),
            "pose2": torch.from_numpy(pose2.astype(np.float32)),
            "bbox2": torch.from_numpy(bbox2.astype(np.float32)),
        }


class SOPatchDataset(Dataset):
    def __init__(
        self,
        subfolder_opt="opt/",
        subfolder_sar="sar/",
        base_path="datasets/SOPatch/OSdataset/",
        train=True,
        preprocessing=None,
        pairs_per_scene=100,
        image_size=512,
    ):
        self.base_path = base_path
        self.train = train
        self.preprocessing = preprocessing
        self.pairs_per_scene = pairs_per_scene
        self.subfolder_opt = subfolder_opt
        self.subfolder_sar = subfolder_sar
        self.image_size = image_size

        self.dataset = []

    def build_dataset(self):
        self.dataset = []

        if not self.train:
            np_random_state = np.random.get_state()
            np.random.seed(42)
            print("Building the validation dataset...")
            mid_path = "val/"
        else:
            print("Building a new training dataset...")
            mid_path = "train/"

        # path = self.base_path + self.subfolder_opt + mid_path + r"*.png"

        imgfiles1 = glob.glob(self.base_path + mid_path + self.subfolder_opt + r"*.png")
        imgfiles2 = glob.glob(self.base_path + mid_path + self.subfolder_sar + r"*.png")

        for i in tqdm(range(len(imgfiles1))):
            image_path1 = imgfiles1[i]
            image_path2 = imgfiles2[i]

            # numpy shape
            self.dataset.append(
                {
                    "image_path1": image_path1,
                    "image_path2": image_path2,
                }
            )

        np.random.shuffle(self.dataset)
        if not self.train:
            np.random.set_state(np_random_state)

    def __len__(self):
        return len(self.dataset)

    def recover_pair(self, pair_metadata):
        # load image to numpy array
        image1 = Image.open(pair_metadata["image_path1"])
        if image1.mode != "RGB":
            image1 = image1.convert("RGB")
        image1 = np.array(image1)

        image2 = Image.open(pair_metadata["image_path2"])
        if image2.mode != "RGB":
            image2 = image2.convert("RGB")
        image2 = np.array(image2)

        return image1, image2

    def init_my_params(self):
        self.depth1 = np.ones((self.image_size, self.image_size))
        self.intrinsics1 = np.identity(3)
        self.pose1 = np.identity(4)
        self.bbox1 = np.array([0, 0])

        self.depth2 = self.depth1
        self.intrinsics2 = self.intrinsics1
        self.pose2 = self.pose1
        self.bbox2 = self.bbox1
        return (
            self.depth1,
            self.intrinsics1,
            self.pose1,
            self.bbox1,
            self.depth2,
            self.intrinsics2,
            self.pose2,
            self.bbox2,
        )

    def __getitem__(self, idx):
        (
            image1,
            image2,
        ) = self.recover_pair(self.dataset[idx])

        image1 = preprocess_image(image1, preprocessing=self.preprocessing)
        image2 = preprocess_image(image2, preprocessing=self.preprocessing)

        (
            depth1,
            intrinsics1,
            pose1,
            bbox1,
            depth2,
            intrinsics2,
            pose2,
            bbox2,
        ) = self.init_my_params()

        return {
            "image1": torch.from_numpy(image1.astype(np.float32)),
            "depth1": torch.from_numpy(depth1.astype(np.float32)),
            "intrinsics1": torch.from_numpy(intrinsics1.astype(np.float32)),
            "pose1": torch.from_numpy(pose1.astype(np.float32)),
            "bbox1": torch.from_numpy(bbox1.astype(np.float32)),
            "image2": torch.from_numpy(image2.astype(np.float32)),
            "depth2": torch.from_numpy(depth2.astype(np.float32)),
            "intrinsics2": torch.from_numpy(intrinsics2.astype(np.float32)),
            "pose2": torch.from_numpy(pose2.astype(np.float32)),
            "bbox2": torch.from_numpy(bbox2.astype(np.float32)),
        }


class WhuDatasetCrop(Dataset):
    def __init__(
        self,
        scene_list_path="datasets_utils/whu-opt-sar-utils/train.txt",
        # scene_info_path="/local/dataset/megadepth/scene_info",
        subfolder_opt="crop/optical/",
        subfolder_sar="crop/sar/",
        base_path="datasets/whu-opt-sar/",
        train=True,
        preprocessing=None,
        # pairs_per_scene=100,
        image_size=256,
    ):
        self.scenes = []
        with open(scene_list_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.scenes.append(line.strip("\n") + ".tif")
        self.base_path = base_path
        self.train = train
        self.preprocessing = preprocessing
        # self.pairs_per_scene = pairs_per_scene
        self.subfolder_opt = subfolder_opt
        self.subfolder_sar = subfolder_sar
        self.image_size = image_size

        self.dataset = []

    def build_dataset(self):
        self.dataset = []

        if not self.train:
            np_random_state = np.random.get_state()
            np.random.seed(42)
            print("Building the validation dataset...")
        else:
            print("Building a new training dataset...")

        for scene in tqdm(self.scenes, total=len(self.scenes)):
            image_path1 = os.path.join(self.base_path, self.subfolder_opt, scene)
            image_path2 = os.path.join(self.base_path, self.subfolder_sar, scene)

            self.dataset.append(
                {
                    "image_path1": image_path1,
                    "image_path2": image_path2,
                }
            )

        np.random.shuffle(self.dataset)
        if not self.train:
            np.random.set_state(np_random_state)

    def __len__(self):
        return len(self.dataset)

    def recover_pair(self, pair_metadata):
        # load image to numpy array
        image1 = Image.open(pair_metadata["image_path1"])
        if image1.mode != "RGB":
            image1 = image1.convert("RGB")
        image1 = np.array(image1)

        image2 = Image.open(pair_metadata["image_path2"])
        if image2.mode != "RGB":
            image2 = image2.convert("RGB")
        image2 = np.array(image2)

        return image1, image2

    def init_my_params(self):
        self.depth1 = np.ones((self.image_size, self.image_size))
        # self.intrinsics1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.intrinsics1 = np.identity(3)
        # pose=[4,4]
        # self.pose1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.pose1 = np.identity(4)
        self.bbox1 = np.array([0, 0])

        self.depth2 = self.depth1
        self.intrinsics2 = self.intrinsics1
        self.pose2 = self.pose1
        self.bbox2 = self.bbox1
        return (
            self.depth1,
            self.intrinsics1,
            self.pose1,
            self.bbox1,
            self.depth2,
            self.intrinsics2,
            self.pose2,
            self.bbox2,
        )

    def __getitem__(self, idx):
        (
            image1,
            image2,
        ) = self.recover_pair(self.dataset[idx])

        image1 = preprocess_image(image1, preprocessing=self.preprocessing)
        image2 = preprocess_image(image2, preprocessing=self.preprocessing)

        (
            depth1,
            intrinsics1,
            pose1,
            bbox1,
            depth2,
            intrinsics2,
            pose2,
            bbox2,
        ) = self.init_my_params()

        return {
            "image1": torch.from_numpy(image1.astype(np.float32)),
            "depth1": torch.from_numpy(depth1.astype(np.float32)),
            "intrinsics1": torch.from_numpy(intrinsics1.astype(np.float32)),
            "pose1": torch.from_numpy(pose1.astype(np.float32)),
            "bbox1": torch.from_numpy(bbox1.astype(np.float32)),
            "image2": torch.from_numpy(image2.astype(np.float32)),
            "depth2": torch.from_numpy(depth2.astype(np.float32)),
            "intrinsics2": torch.from_numpy(intrinsics2.astype(np.float32)),
            "pose2": torch.from_numpy(pose2.astype(np.float32)),
            "bbox2": torch.from_numpy(bbox2.astype(np.float32)),
        }


class OSDataset(Dataset):
    def __init__(
        self,
        scene_list_path="datasets_utils/osdataset/train.txt",
        # scene_info_path="/local/dataset/megadepth/scene_info",
        # subfolder_opt="datasets/OSdataset/512/train",
        # subfolder_sar="datasets/OSdataset/512/train",
        base_path="datasets/OSdataset/512/",
        train=True,
        preprocessing=None,
        # pairs_per_scene=100,
        image_size=512,
    ):
        self.scenes = []
        with open(scene_list_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.scenes.append(line.strip("\n") + ".png")
        self.base_path = base_path
        self.base_path_2 = ""
        self.train = train
        self.preprocessing = preprocessing
        # self.pairs_per_scene = pairs_per_scene
        # self.subfolder_opt = subfolder_opt
        # self.subfolder_sar = subfolder_sar
        self.image_size = image_size

        self.dataset = []

    def build_dataset(self):
        self.dataset = []

        if not self.train:
            np_random_state = np.random.get_state()
            np.random.seed(42)
            self.base_path_2 = os.path.join(self.base_path, "test")
            print("Building the validation dataset...")
        else:
            self.base_path_2 = os.path.join(self.base_path, "train")
            print("Building a new training dataset...")

        for scene in tqdm(self.scenes, total=len(self.scenes)):
            image_path1 = os.path.join(self.base_path_2, "sar" + scene)
            image_path2 = os.path.join(self.base_path_2, "opt" + scene)

            self.dataset.append(
                {
                    "image_path1": image_path1,
                    "image_path2": image_path2,
                }
            )

        np.random.shuffle(self.dataset)
        if not self.train:
            np.random.set_state(np_random_state)

    def __len__(self):
        return len(self.dataset)

    def recover_pair(self, pair_metadata):
        # load image to numpy array
        image1 = Image.open(pair_metadata["image_path1"])
        if image1.mode != "RGB":
            image1 = image1.convert("RGB")
        image1 = np.array(image1)

        image2 = Image.open(pair_metadata["image_path2"])
        if image2.mode != "RGB":
            image2 = image2.convert("RGB")
        image2 = np.array(image2)

        return image1, image2

    def init_my_params(self):
        self.depth1 = np.ones((self.image_size, self.image_size))
        # self.intrinsics1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.intrinsics1 = np.identity(3)
        # pose=[4,4]
        # self.pose1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.pose1 = np.identity(4)
        self.bbox1 = np.array([0, 0])

        self.depth2 = self.depth1
        self.intrinsics2 = self.intrinsics1
        self.pose2 = self.pose1
        self.bbox2 = self.bbox1
        return (
            self.depth1,
            self.intrinsics1,
            self.pose1,
            self.bbox1,
            self.depth2,
            self.intrinsics2,
            self.pose2,
            self.bbox2,
        )

    def __getitem__(self, idx):
        (
            image1,
            image2,
        ) = self.recover_pair(self.dataset[idx])

        image1 = preprocess_image(image1, preprocessing=self.preprocessing)
        image2 = preprocess_image(image2, preprocessing=self.preprocessing)

        (
            depth1,
            intrinsics1,
            pose1,
            bbox1,
            depth2,
            intrinsics2,
            pose2,
            bbox2,
        ) = self.init_my_params()

        return {
            "image1": torch.from_numpy(image1.astype(np.float32)),
            "depth1": torch.from_numpy(depth1.astype(np.float32)),
            "intrinsics1": torch.from_numpy(intrinsics1.astype(np.float32)),
            "pose1": torch.from_numpy(pose1.astype(np.float32)),
            "bbox1": torch.from_numpy(bbox1.astype(np.float32)),
            "image2": torch.from_numpy(image2.astype(np.float32)),
            "depth2": torch.from_numpy(depth2.astype(np.float32)),
            "intrinsics2": torch.from_numpy(intrinsics2.astype(np.float32)),
            "pose2": torch.from_numpy(pose2.astype(np.float32)),
            "bbox2": torch.from_numpy(bbox2.astype(np.float32)),
        }
