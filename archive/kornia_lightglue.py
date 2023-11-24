import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import torch
from kornia_moons.feature import draw_LAF_matches

device = K.utils.get_cuda_device_if_available()
disk = K.feature.DISK.from_pretrained("depth").to(device)
lightglue = K.feature.LightGlueMatcher("disk").to(device)
num_features = 2048

fname1 = r"D:\Code\LoFTR\20231122162112.jpg"
fname2 = r"D:\Code\LoFTR\20231122162117.jpg"

# fname1 = r"datasets/SOPatch/OSdataset/test/opt/d20002.png"
# fname2 = r"datasets/SOPatch/OSdataset/test/sar/d20002.png"

img1 = K.io.load_image(fname1, K.io.ImageLoadType.RGB32, device=device)[None, ...]
img2 = K.io.load_image(fname2, K.io.ImageLoadType.RGB32, device=device)[None, ...]

# 推理模式
with torch.inference_mode():
    inp = torch.cat([img1, img2], dim=0)
    features1, features2 = disk(inp, num_features, pad_if_not_divisible=True)
    # 2048*2 2048*128
    kps1, descs1 = features1.keypoints, features1.descriptors
    kps2, descs2 = features2.keypoints, features2.descriptors

    # 使用kornia.feature模块中的laf_from_center_scale_ori函数，根据关键点的位置、尺度和方向生成局部仿射框架（Local Affine Frames，LAFs）
    lafs1 = KF.laf_from_center_scale_ori(
        kps1[None], torch.ones(1, len(kps1), 1, 1, device=device)
    )
    lafs2 = KF.laf_from_center_scale_ori(
        kps2[None], torch.ones(1, len(kps2), 1, 1, device=device)
    )

    dists, idxs = lightglue(descs1, descs2, lafs1, lafs2)

print(f"{idxs.shape[0]} tentative matches with DISK lightglue")


def get_matching_keypoints(kp1, kp2, idxs):
    mkpts1 = kp1[idxs[:, 0]]
    mkpts2 = kp2[idxs[:, 1]]
    return mkpts1, mkpts2


mkpts1, mkpts2 = get_matching_keypoints(kps1, kps2, idxs)

Fm, inliers = cv2.findHomography(
    mkpts1.detach().cpu().numpy(),
    mkpts2.detach().cpu().numpy(),
    cv2.USAC_MAGSAC,
    1.0,
    0.999,
    100000,
)
inliers = inliers > 0
print(f"{inliers.sum()} inliers with DISK")


# # 最后，让我们使用kornia_moons的函数绘制匹配图像。正确的匹配用绿色表示，不精确的匹配用蓝色表示
fig, ax = draw_LAF_matches(
    KF.laf_from_center_scale_ori(kps1[None].cpu()),
    KF.laf_from_center_scale_ori(kps2[None].cpu()),
    idxs.cpu(),
    K.tensor_to_image(img1.cpu()),
    K.tensor_to_image(img2.cpu()),
    inliers,
    draw_dict={
        "inlier_color": (0.2, 1, 0.2),
        "tentative_color": (1, 1, 0.2, 0.3),
        "feature_color": None,
        "vertical": False,
    },
    return_fig_ax=True,
)
# # 全屏显示

# 全屏显示
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()
