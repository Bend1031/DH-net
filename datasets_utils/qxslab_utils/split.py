import os
import random

# Path to the directory containing the PNG files
png_dir = "../QXSLAB_SAROPT/opt_256_oc_0.2"

# Get a list of all the PNG files in the directory
png_files = [f for f in os.listdir(png_dir) if f.endswith(".png")]

# Shuffle the list of PNG files
random.seed(1)
random.shuffle(png_files)


# 去除png后缀
png_files = [f[:-4] for f in png_files]

num_files = int(len(png_files) * 0.8)
valid_png_files = png_files[num_files:]
# Take only 80% of the shuffled PNG files
png_files = png_files[:num_files]


# Write the selected PNG file names to a file
with open("train.txt", "w") as f:
    for png_file in png_files:
        f.write(os.path.basename(png_file) + "\n")

# the rest of the PNG files are used for validation
png_files = valid_png_files
with open("valid.txt", "w") as f:
    for png_file in png_files:
        f.write(os.path.basename(png_file) + "\n")
