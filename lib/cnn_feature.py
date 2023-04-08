import numpy as np
import scipy
import scipy.io
import scipy.misc
import torch
from PIL import Image

from lib.model_test import D2Net
from lib.pyramid import process_multiscale
from lib.utils import preprocess_image

use_cuda = torch.cuda.is_available()

# Creating CNN model
# model = D2Net(model_file="models/d2.00.pth", use_cuda=use_cuda)
model = D2Net(model_file="checkpoints/d2.18.pth", use_cuda=use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")

multiscale = False
max_edge = 2500
max_sum_edges = 5000


def resize_image_with_pil(image, scale_factor):
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    input_pil_image = Image.fromarray(image.astype("uint8"))
    resized_image = input_pil_image.resize((new_width, new_height), Image.ANTIALIAS)
    return np.array(resized_image).astype("float")


# de-net feature extract function
def cnn_feature_extract(image, scales=[0.25, 0.50, 1.0], nfeatures=1000):
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
