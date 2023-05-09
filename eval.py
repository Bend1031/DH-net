import os

import imageio
import numpy as np
import torch
from PIL import Image
from skimage.measure import ransac
from skimage.transform import AffineTransform
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# from lib.dataset import QxslabSarOptDataset
from lib.eval_match import flann
from lib.model_test import D2Net
from lib.pyramid import process_multiscale
from lib.utils import preprocess_image


class QxslabSarOptDataset(Dataset):
    def __init__(
        self,
        scene_list_path="qxslab_utils/train.txt",
        base_path="datasets/QXSLAB_SAROPT/",
        subfolder_opt="opt_256_oc_0.2/",
        subfolder_sar="sar_256_oc_0.2/",
    ):
        self.scenes = []
        with open(scene_list_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.scenes.append(line.strip("\n") + ".png")

        # self.scene_list_path = scene_list_path
        self.base_path = base_path

        self.subfolder_opt = subfolder_opt
        self.subfolder_sar = subfolder_sar

        self.dataset = []
        self.build_dataset()

    def build_dataset(self):
        self.dataset = []

        for scene in tqdm(self.scenes, total=len(self.scenes)):
            image_path1 = os.path.join(self.base_path, self.subfolder_opt, scene)
            image_path2 = os.path.join(self.base_path, self.subfolder_sar, scene)

            self.dataset.append(
                {
                    "image_path1": image_path1,
                    "image_path2": image_path2,
                }
            )

    def recover_pair(self, pair_metadata):
        image1 = imageio.v3.imread(pair_metadata["image_path1"])
        image2 = imageio.v3.imread(pair_metadata["image_path2"])
        # asarray
        image1 = np.array(image1)
        image2 = np.array(image2)
        # image1
        return image1, image2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (
            image1,
            image2,
        ) = self.recover_pair(self.dataset[idx])

        return {
            "image1": image1,
            "image2": image2,
        }


def prepare_data(data_path, batch_size=1):
    dataset = QxslabSarOptDataset(data_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return dataloader


def resize_image_with_pil(image, scale_factor):
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    input_pil_image = Image.fromarray(image.astype("uint8"))
    resized_image = input_pil_image.resize((new_width, new_height), Image.ANTIALIAS)
    return np.array(resized_image).astype("float")


def evaluate(model, device, dataloader):
    model.eval()
    accuracy = []
    # with torch.no_grad():
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, batch in progress_bar:
        # tensor  [1,256,256,3]->[256,256,3]
        image1 = batch["image1"].numpy()[0]
        image2 = batch["image2"].numpy()[0]
        kps_left, sco_left, des_left = cnn_feature_extract(
            image1, model, device, nfeatures=-1
        )
        kps_right, sco_right, des_right = cnn_feature_extract(
            image2, model, device, nfeatures=-1
        )

        locations_1_to_use, locations_2_to_use = flann(
            kps_left, des_left, kps_right, des_right
        )

        # %% Perform geometric verification using RANSAC.
        _, inliers = ransac(
            (locations_1_to_use, locations_2_to_use),
            AffineTransform,
            min_samples=4,
            residual_threshold=4,
            max_trials=1000,
        )

        accuracy.append(sum(inliers) / len(locations_1_to_use))
        progress_bar.set_postfix(acc=f"{np.mean(accuracy):.4f}")
    return np.mean(accuracy)


def cnn_feature_extract(image, model, device, scales=[0.25, 0.50, 1.0], nfeatures=1000):
    model.eval()
    multiscale = False
    max_edge = 2500
    max_sum_edges = 5000
    # repeat single channel image to 3 channel
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # Resize image to maximum size.
    resized_image = image

    # 如果最大边大于max_edge，则调整大小
    if max(resized_image.shape) > max_edge:
        scale_factor = max_edge / max(resized_image.shape)
        resized_image = resize_image_with_pil(resized_image, scale_factor)

    # 如果尺寸之和大于max_sum_edges，则调整大小
    if sum(resized_image.shape[:2]) > max_sum_edges:
        scale_factor = max_sum_edges / sum(resized_image.shape[:2])
        resized_image = resize_image_with_pil(resized_image, scale_factor)

    # resize proportion
    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(resized_image, preprocessing="torch")
    with torch.no_grad():
        # Process image with D2-Net
        if multiscale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32), device=device
                ),
                model,
                scales,
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32), device=device
                ),
                model,
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
        keypoints = res[0:nfeatures, 1:4].copy()
        descriptors = res[0:nfeatures, 4:].copy()
        del res
    return keypoints, scores, descriptors


def main():
    data_path = "datasets_utils/qxslab_utils/train/valid.txt"
    model_path = "checkpoints/qxslab/qxs.18.pth"
    # model_path = "models/d2_t f.pth"

    use_cuda = torch.cuda.is_available()
    model = D2Net(model_file=model_path, use_cuda=use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    dataloader = prepare_data(data_path)
    accuracy = evaluate(model, device, dataloader)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
