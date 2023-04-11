# 读取 datasets/whu-opt-sar/crop 下的tif文件，按照比例划分为训练集和验证集
# 保存到 datasets_utils/whu-opt-sar-utils/train.txt 和 datasets_utils/whu-opt-sar-utils/valid.txt


import os
import random

# Path to the directory containing the PNG files
tif_dir = r"D:\Code\d2-net\datasets\whu-opt-sar\crop\optical"

# Get a list of all the PNG files in the directory
tif_files = [f for f in os.listdir(tif_dir) if f.endswith(".tif")]

# Shuffle the list of PNG files
random.seed(1)
random.shuffle(tif_files)


# 去除png后缀
tif_files = [f[:-4] for f in tif_files]

num_files = int(len(tif_files) * 0.8)
valid_tif_files = tif_files[num_files:]
# Take only 80% of the shuffled PNG files
tif_files = tif_files[:num_files]


# Write the selected PNG file names to a file
with open("train.txt", "w") as f:
    for tif_file in tif_files:
        f.write(os.path.basename(tif_file) + "\n")

# the rest of the PNG files are used for validation
tif_files = valid_tif_files
with open("valid.txt", "w") as f:
    for tif_file in tif_files:
        f.write(os.path.basename(tif_file) + "\n")
