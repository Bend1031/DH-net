import glob
import logging
import os
import time

import cv2
import hydra
import numpy as np
from tqdm import tqdm

from components import load_component
from lib.config import Config
from lib.rootpath import rootPath
from lib.utils import pix2pix_RMSE
from utils import evaluation_utils

# file_handler = logging.FileHandler("./log.txt", encoding="utf-8")
# logging.basicConfig(level=logging.INFO, handlers={file_handler})
log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="eval_sp_sg",
)
def main(config: Config):
    dataset = config.dataset.dataset_path
    log.info(f"数据集为{dataset}")
    # print(1)


if __name__ == "__main__":
    main()
