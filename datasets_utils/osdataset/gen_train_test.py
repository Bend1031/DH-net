import os
import sys

png_dir = r"D:\Code\d2-net\datasets\OSdataset\512\train"

# 读取所有png文件
png_files = [f for f in os.listdir(png_dir) if f.endswith(".png")]
# 分为opt与sar
opt_files = [f for f in png_files if f.startswith("opt")]
# sar_files = [f for f in png_files if f.startswith("sar")]
# 只留下文件名中的数字
opt_files = [f.split(".")[0].split("opt")[1] for f in opt_files]
# sar_files = [f.split(".")[0].split("_")[1] for f in sar_files]
# print(opt_files)

# 生成txt
with open(r"D:\Code\d2-net\datasets_utils\osdataset\train.txt", "w") as f:
    for f_name in opt_files:
        f.write(f_name + "\n")

png_dir = r"D:\Code\d2-net\datasets\OSdataset\512\test"

# 读取所有png文件
png_files = [f for f in os.listdir(png_dir) if f.endswith(".png")]
# 分为opt与sar
opt_files = [f for f in png_files if f.startswith("opt")]
# sar_files = [f for f in png_files if f.startswith("sar")]
# 只留下文件名中的数字
opt_files = [f.split(".")[0].split("opt")[1] for f in opt_files]
# sar_files = [f.split(".")[0].split("_")[1] for f in sar_files]
# print(opt_files)

# 生成txt
with open(r"D:\Code\d2-net\datasets_utils\osdataset\test.txt", "w") as f:
    for f_name in opt_files:
        f.write(f_name + "\n")
