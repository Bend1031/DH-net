# %%
import glob
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plottable import Table

# 默认颜色和线型
default_color = "black"
default_linestyle = "-"


def load_errors():
    err_list = []
    jsonlist = glob.glob("result/*.json")
    for jsonfile in jsonlist:
        with open(jsonfile, "r") as f:
            err = json.load(f)
        err_list.append(err)
    return err_list


def plot_curves(keys, datasets, err_list, color_dict, linestyle_dict):
    for key in keys:
        plt_lim = [1, 10]
        plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)
        plt.rc("axes", titlesize=25)
        plt.rc("axes", labelsize=25)
        plt.figure(figsize=(15, 5))

        for idx, dataset in enumerate(datasets, start=1):
            plt.subplot(1, 3, idx)
            for i, err in enumerate(err_list):
                if err["dataset"] == dataset:
                    name = err["method"]
                    ls = linestyle_dict.get(name, default_linestyle)
                    color = color_dict.get(name, default_color)
                    error = err[key]

                    if not error:
                        continue

                    plt.plot(
                        plt_rng,
                        [error[str(thr)] for thr in plt_rng],
                        color=color,
                        ls=ls,
                        linewidth=3,
                        label=name,
                    )

            plt.title(dataset)
            plt.xlim(plt_lim)
            plt.xticks(plt_rng)
            plt.ylim([0, 1])

            if idx > 1:
                plt.gca().axes.set_yticklabels([])
            if idx == 1:
                plt.ylabel("MMA")

            if idx == 2:
                plt.xlabel("threshold [px]")
                plt.legend()
            plt.grid()
            plt.tick_params(axis="both", which="major", labelsize=20)

        plt.tight_layout()
        plt.savefig(f"result/img/{key}.png", bbox_inches="tight", dpi=300)
        plt.show()

    print("Done!")


def plot_time_histogram(dataset, err_list, color_dict):
    method_list = []
    time_list = []

    for item in err_list:
        if item["dataset"] == dataset:
            method_list.append(item["method"])
            time_list.append(int(item["Time"][:-1]))

    time_list, method_list = zip(*sorted(zip(time_list, method_list)))
    color_list = [color_dict.get(item, default_color) for item in method_list]

    plt.barh(method_list, time_list, color=color_list)
    plt.title(dataset)
    plt.tick_params(axis="both", which="major", labelsize=20)


def plot_tables(datasets, err_list):
    mma_list = [f"MMA@{i}" for i in range(3, 10, 3)]
    column = ["Dataset", "Methods"] + mma_list

    for dataset in datasets:
        data = []
        for err in err_list:
            if dataset == err["dataset"]:
                name = err["method"]
                error = err["homo_mma"]
                if not error:
                    continue
                data.append(
                    [dataset]
                    + [name]
                    + [error[str(thr)] * 100 for thr in range(3, 10, 3)]
                )

        df = pd.DataFrame(data=data, columns=column).round(4)
        df = df.sort_values("MMA@9")
        df.set_index("Dataset", inplace=True)
        grouped = df.groupby("Dataset")

        plt.figure()
        for name, group in grouped:
            Table(
                group,
                textprops={
                    "fontsize": 10,
                    "fontfamily": "Times New Roman",
                    "ha": "center",
                    "va": "center",
                },
                col_label_divider=True,
                footer_divider=True,
                row_dividers=False,
            )
            plt.show()
            group.to_excel(f"result/table/{name}.xlsx")


def plot_time_comparison(datasets, err_list, color_dict):
    plt.rc("axes", titlesize=25)
    plt.rc("axes", labelsize=25)
    plt.figure(figsize=(15, 5))

    for idx, dataset in enumerate(datasets, start=1):
        plt.subplot(1, 3, idx)
        plot_time_histogram(dataset, err_list, color_dict)
        if idx == 2:
            plt.xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(f"result/img/time_histogram.png", bbox_inches="tight", dpi=300)
    plt.show()

    print("Done!")


def plot_performance_tables(datasets, err_list):
    column = ["Methods", "NCM", "CMR", "RMSE", "SR"]

    for dataset in datasets:
        data = []
        for err in err_list:
            name = err["method"]
            if dataset == err["dataset"]:
                ncm = err["NCM"]
                cmr = err["CMR"]
                rmse = err["RMSE"]
                sr = err["success_rate"]
                data.append([name] + [ncm] + [cmr] + [rmse] + [sr])

        df = pd.DataFrame(data=data, columns=column).round(4)
        df = df.sort_values("SR", ascending=True)
        df.set_index("Methods", inplace=True)
        plt.figure(figsize=(5, 5))
        Table(
            df,
            textprops={
                "fontsize": 10,
                "fontfamily": "Times New Roman",
                "ha": "center",
                "va": "center",
            },
            col_label_divider=True,
            footer_divider=True,
            row_dividers=False,
        )
        plt.rc("axes", titlesize=15)
        plt.title(dataset)
        plt.tight_layout()
        plt.savefig(f"result/img/{dataset}_table.png", bbox_inches="tight", dpi=300)
        plt.show()


#%% 主程序
err_list = load_errors()

color_dict = {
    "SP-SG": "red",
    "D2-BF": "green",
    "SP-SGM": "orange",
    "CMM": "blue",
    "D2": "purple",
    "DISK-LG": "cyan",
    "SP-LG": "brown",
    "LoFTR": "yellow",
}

linestyle_dict = {"sp_SG_magsacpp": "-", "d2_BF_magsacpp": "-"}

keys = ["corr_mma", "ransac_mma", "homo_mma"]
datasets = ["osdataset", "sen1-2", "whu-sen-city"]

# 绘制曲线图
plot_curves(keys, datasets, err_list, color_dict, linestyle_dict)

# 绘制时间对比图
plot_time_comparison(datasets, err_list, color_dict)

# 绘制性能表格
plot_performance_tables(datasets, err_list)

# %%
