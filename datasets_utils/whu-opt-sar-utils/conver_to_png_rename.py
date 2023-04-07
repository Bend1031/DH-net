import os

import cv2
from tqdm import tqdm


def convert_to_png_rename(img_path, img_png_path):
    # create path if it doesn't exist
    if not os.path.exists(img_png_path):
        os.mkdir(img_png_path)

    for image in tqdm(os.listdir(img_path)):
        # load image
        # img = cv2.imread(os.path.join(sar_path, image), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(os.path.join(img_path, image))
        # convert to png
        cv2.imwrite(os.path.join(img_png_path, image[:-4] + ".png"), img)


if __name__ == "__main__":
    dataset_path = r"D:\Code\d2-net\datasets\whu-opt-sar"
    sar_path = os.path.join(dataset_path, "sar")
    # path to SAR png images
    sar_png_path = os.path.join(dataset_path, "sar_png")

    # optical
    opt_path = os.path.join(dataset_path, "opt")
    # path to optical png images
    opt_png_path = os.path.join(dataset_path, "opt_png")

    convert_to_png_rename(sar_path, sar_png_path)
    convert_to_png_rename(opt_path, opt_png_path)
