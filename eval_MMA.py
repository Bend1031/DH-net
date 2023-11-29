"""生成MMA曲线图"""

import glob
import json

import matplotlib.pyplot as plt
import numpy as np

# 读取方法名、dataset与err
err_list = []
jsonlist = glob.glob("result/*.json")
for jsonfile in jsonlist:
    with open(jsonfile, "r") as f:
        err = json.load(f)
    err_list.append(err)


color_dict = {
    "SP-SG": "red",
    "D2-BF": "green",
    "SP-SGM": "orange",
    "CMM": "blue",
    "D2": "purple",
    "DISK-LightGlue": "cyan",
    "SP-LightGlue": "brown",
    "LoFTR": "yellow"
    # 添加更多方法名和颜色的映射
}

linestyle_dict = {
    "sp_SG_magsacpp": "-",
    "d2_BF_magsacpp": "-",
    # 添加更多方法名和颜色的映射
}

# keys = ["corr_mma", "ransac_mma", "homo_mma"]

# for key in keys:
#     # 设置全局参数
#     plt_lim = [1, 10]
#     plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)
#     plt.rc("axes", titlesize=25)
#     plt.rc("axes", labelsize=25)

#     # 创建子图
#     plt.figure(figsize=(15, 5))

#     plt.subplot(1, 3, 1)
#     for i, err in enumerate(err_list):
#         """绘制MMA曲线"""
#         dataset = err["dataset"]
#         if dataset == "osdataset":
#             name = err["method"]
#             ls = linestyle_dict.get(name, "-")
#             color = color_dict.get(name, "black")
#             error = err[key]
#             if error == {}:
#                 continue
#             plt.plot(
#                 plt_rng,
#                 [error[str(thr)] for thr in plt_rng],
#                 color=color,
#                 ls=ls,
#                 linewidth=3,
#                 label=name,
#             )
#     plt.title("osdataset")
#     plt.xlim(plt_lim)
#     plt.xticks(plt_rng)
#     plt.ylabel("MMA")
#     plt.ylim([0, 1])
#     plt.grid()
#     plt.tick_params(axis="both", which="major", labelsize=20)
#     plt.legend()

#     plt.subplot(1, 3, 2)
#     for i, err in enumerate(err_list):
#         """绘制MMA曲线"""
#         dataset = err["dataset"]
#         if dataset == "sen1-2":
#             name = err["method"]
#             ls = linestyle_dict.get(name, "-")
#             color = color_dict.get(name, "black")
#             error = err[key]
#             if error == {}:
#                 continue
#             plt.plot(
#                 plt_rng,
#                 [error[str(thr)] for thr in plt_rng],
#                 color=color,
#                 ls=ls,
#                 linewidth=3,
#                 label=name,
#             )
#     plt.title("sen1-2")
#     plt.xlabel("threshold [px]")
#     plt.xlim(plt_lim)
#     plt.xticks(plt_rng)
#     plt.ylim([0, 1])
#     plt.gca().axes.set_yticklabels([])
#     plt.grid()
#     plt.tick_params(axis="both", which="major", labelsize=20)
#     plt.legend()

#     plt.subplot(1, 3, 3)
#     for i, err in enumerate(err_list):
#         """绘制MMA曲线"""
#         dataset = err["dataset"]
#         if dataset == "whu-sen-city":
#             name = err["method"]
#             ls = linestyle_dict.get(name, "-")
#             color = color_dict.get(name, "black")
#             error = err[key]
#             if error == {}:
#                 continue
#             plt.plot(
#                 plt_rng,
#                 [error[str(thr)] for thr in plt_rng],
#                 color=color,
#                 ls=ls,
#                 linewidth=3,
#                 label=name,
#             )
#     plt.title("whu-sen-city")
#     plt.xlim(plt_lim)
#     plt.xticks(plt_rng)
#     plt.ylim([0, 1])
#     plt.gca().axes.set_yticklabels([])
#     plt.grid()
#     plt.tick_params(axis="both", which="major", labelsize=20)
#     plt.legend()

#     # 保存并显示图像
#     plt.tight_layout()
#     plt.savefig(f"result_img/{key}.png", bbox_inches="tight", dpi=300)
#     plt.show()

# print("Done!")

time_dict = {key: [] for key in color_dict.keys()}  # 新增一个字典用于存储各方法的匹配时间

jsonlist = glob.glob("result/*.json")
for jsonfile in jsonlist:
    with open(jsonfile, "r") as f:
        err = json.load(f)

    method_name = err["method"]
    matching_time = float(err["Time"][:-1])  # 提取匹配时间，去除末尾的's'并转为float
    time_dict[method_name].append(matching_time)
# 创建子图
plt_lim = [1, 10]
plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)
plt.rc("axes", titlesize=25)
plt.rc("axes", labelsize=25)

# 创建子图
plt.figure(figsize=(15, 5))

for idx, dataset_name in enumerate(["osdataset", "sen1-2", "whu-sen-city"]):
    plt.subplot(1, 3, idx + 1)
    for method_name, times in time_dict.items():
        dataset = [err["dataset"] for err in err_list if err["method"] == method_name][
            0
        ]
        if dataset == dataset_name:
            plt.bar(
                method_name, np.mean(times), color=color_dict.get(method_name, "black")
            )

    plt.title(dataset_name)
    plt.ylabel("Matching Time (s)")
    plt.grid()
    plt.tick_params(axis="both", which="major", labelsize=20)

# 保存并显示图像
plt.tight_layout()
plt.savefig(f"result_img/matching_time.png", bbox_inches="tight", dpi=300)
plt.show()
