import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.exceptions import EmptyTensorError
from lib.utils import interpolate_dense_features, upscale_positions


def process_multiscale(image, model, scales=[0.5, 1, 2]):
    # 获取输入图像的形状信息
    b, _, h_init, w_init = image.size()
    device = image.device
    assert b == 1  # 确保批量大小为1

    # 初始化存储关键点、描述符和分数的张量
    all_keypoints = torch.zeros([3, 0])
    all_descriptors = torch.zeros([model.dense_feature_extraction.num_channels, 0])
    all_scores = torch.zeros(0)

    previous_dense_features = None
    banned = None

    # 遍历不同尺度
    for idx, scale in enumerate(scales):
        # 对输入图像进行插值得到当前尺度的图像
        current_image = F.interpolate(
            image,
            scale_factor=scale,
            recompute_scale_factor=True,
            mode="bilinear",
            align_corners=True,
        )
        _, _, h_level, w_level = current_image.size()

        # 使用模型提取密集特征
        dense_features = model.dense_feature_extraction(current_image)
        del current_image

        _, _, h, w = dense_features.size()

        # 将前一尺度的特征图上采样并与当前尺度的特征图相加
        if previous_dense_features is not None:
            dense_features += F.interpolate(
                previous_dense_features,
                size=[h, w],
                mode="bilinear",
                align_corners=True,
            )
            del previous_dense_features

        # 使用检测模块获取检测结果
        detections = model.detection(dense_features)
        if banned is not None:
            banned = F.interpolate(banned.float(), size=[h, w]).bool()
            detections = torch.min(detections, ~banned)
            banned = torch.max(torch.max(detections, dim=1)[0].unsqueeze(1), banned)
        else:
            banned = torch.max(detections, dim=1)[0].unsqueeze(1)
        fmap_pos = torch.nonzero(detections[0].cpu()).t()
        del detections

        # 使用定位模块获取位移信息
        displacements = model.localization(dense_features)[0].cpu()
        displacements_i = displacements[
            0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ]
        displacements_j = displacements[
            1, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ]
        del displacements

        # 筛选有效的位移
        mask = torch.min(
            torch.abs(displacements_i) < 0.5, torch.abs(displacements_j) < 0.5
        )
        fmap_pos = fmap_pos[:, mask]
        valid_displacements = torch.stack(
            [displacements_i[mask], displacements_j[mask]], dim=0
        )
        del mask, displacements_i, displacements_j

        # 计算特征点在原图上的坐标
        fmap_keypoints = fmap_pos[1:, :].float() + valid_displacements
        del valid_displacements

        try:
            # 插值得到密集特征点对应的原始特征和描述符
            raw_descriptors, _, ids = interpolate_dense_features(
                fmap_keypoints.to(device), dense_features[0]
            )
        except EmptyTensorError:
            continue
        fmap_pos = fmap_pos[:, ids]
        fmap_keypoints = fmap_keypoints[:, ids]
        del ids

        # 对特征点坐标进行上采样
        keypoints = upscale_positions(fmap_keypoints, scaling_steps=2)
        del fmap_keypoints

        # 归一化描述符
        descriptors = F.normalize(raw_descriptors, dim=0).cpu()
        del raw_descriptors

        # 将特征点坐标映射回原图
        keypoints[0, :] *= h_init / h_level
        keypoints[1, :] *= w_init / w_level

        fmap_pos = fmap_pos.cpu()
        keypoints = keypoints.cpu()

        # 将尺度信息添加到特征点坐标中
        keypoints = torch.cat(
            [
                keypoints,
                torch.ones([1, keypoints.size(1)]) * 1 / scale,
            ],
            dim=0,
        )

        # 计算特征点的分数
        scores = dense_features[
            0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ].cpu() / (idx + 1)
        del fmap_pos

        # 将结果累加到总的关键点、描述符和分数中
        all_keypoints = torch.cat([all_keypoints, keypoints], dim=1)
        all_descriptors = torch.cat([all_descriptors, descriptors], dim=1)
        all_scores = torch.cat([all_scores, scores], dim=0)
        del keypoints, descriptors

        previous_dense_features = dense_features
        del dense_features
    del previous_dense_features, banned

    # 将结果转换为NumPy数组并返回
    keypoints = all_keypoints.t().numpy()
    del all_keypoints
    scores = all_scores.numpy()
    del all_scores
    descriptors = all_descriptors.t().numpy()
    del all_descriptors
    return keypoints, scores, descriptors
