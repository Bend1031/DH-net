import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from lib.exceptions import EmptyTensorError, NoGradientError
from lib.utils import (
    downscale_positions,
    grid_positions,
    imshow_image,
    savefig,
    upscale_positions,
)
from utils.pytorch_geom import get_dist_mat, interpolate, rnd_sample


def loss_function(
    model, batch, device, margin=1, safe_radius=4, scaling_steps=3, plot=False
):
    output = model(
        {"image1": batch["image1"].to(device), "image2": batch["image2"].to(device)}
    )

    loss = torch.tensor(np.array([0], dtype=np.float32), device=device)
    has_grad = False

    n_valid_samples = 0
    for idx_in_batch in range(batch["image1"].size(0)):
        # Annotations
        depth1 = batch["depth1"][idx_in_batch].to(device)  # [h1, w1]
        intrinsics1 = batch["intrinsics1"][idx_in_batch].to(device)  # [3, 3]
        pose1 = batch["pose1"][idx_in_batch].view(4, 4).to(device)  # [4, 4]
        bbox1 = batch["bbox1"][idx_in_batch].to(device)  # [2]

        depth2 = batch["depth2"][idx_in_batch].to(device)
        intrinsics2 = batch["intrinsics2"][idx_in_batch].to(device)
        pose2 = batch["pose2"][idx_in_batch].view(4, 4).to(device)
        bbox2 = batch["bbox2"][idx_in_batch].to(device)

        # Network output
        dense_features1 = output["dense_features1"][idx_in_batch]
        #512,64,64
        c, h1, w1 = dense_features1.size()
        scores1 = output["scores1"][idx_in_batch].view(-1)

        dense_features2 = output["dense_features2"][idx_in_batch]
        _, h2, w2 = dense_features2.size()
        scores2 = output["scores2"][idx_in_batch]

        all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)
        descriptors1 = all_descriptors1  # 512*1024

        all_descriptors2 = F.normalize(dense_features2.view(c, -1), dim=0)

        # Warp the positions from image 1 to image 2
        fmap_pos1 = grid_positions(
            h1, w1, device
        )  # 特征图的位置[2, h1*w1]: [[0,0,0,...,w1-1,w1-1,w1-1,...], [0,1,2,...,0,1,2,...]]
        pos1 = upscale_positions(
            fmap_pos1, scaling_steps=scaling_steps
        )  # 原图的位置[2, h1*w1]
        try:
            pos1, pos2, ids = warp(
                pos1,
                depth1,
                intrinsics1,
                pose1,
                bbox1,
                depth2,
                intrinsics2,
                pose2,
                bbox2,
            )
        except EmptyTensorError:
            continue
        fmap_pos1 = fmap_pos1[:, ids]
        descriptors1 = descriptors1[:, ids]
        scores1 = scores1[ids]

        # Skip the pair if not enough GT correspondences are available
        if ids.size(0) < 128:
            continue

        # Descriptors at the corresponding positions
        fmap_pos2 = torch.round(
            downscale_positions(pos2, scaling_steps=scaling_steps)
        ).long()
        descriptors2 = F.normalize(
            dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]], dim=0
        )
        # 4096
        positive_distance = (
            2
            - 2
            * (descriptors1.t().unsqueeze(1) @ descriptors2.t().unsqueeze(2)).squeeze()
        )
        # n匹配的特征点的距离，[n]为n个匹配点的距离
        # 余弦相似度转换为距离度量，这里使用 2 - 2 * 余弦相似度 的形式，将距离范围转换为0到2之间，0表示完全相似，2表示完全不相似

        all_fmap_pos2 = grid_positions(h2, w2, device)
        position_distance = torch.max(
            torch.abs(fmap_pos2.unsqueeze(2).float() - all_fmap_pos2.unsqueeze(1)),
            dim=0,
        )[0]

        is_out_of_safe_radius = position_distance > safe_radius
        distance_matrix = 2 - 2 * (descriptors1.t() @ all_descriptors2)
        negative_distance2 = torch.min(
            distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.0, dim=1
        )[0]

        all_fmap_pos1 = grid_positions(h1, w1, device)
        position_distance = torch.max(
            torch.abs(fmap_pos1.unsqueeze(2).float() - all_fmap_pos1.unsqueeze(1)),
            dim=0,
        )[0]
        is_out_of_safe_radius = position_distance > safe_radius
        distance_matrix = 2 - 2 * (descriptors2.t() @ all_descriptors1)
        negative_distance1 = torch.min(
            distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.0, dim=1
        )[0]

        diff = positive_distance - torch.min(negative_distance1, negative_distance2)

        scores2 = scores2[fmap_pos2[0, :], fmap_pos2[1, :]]

        # loss = loss + (torch.sum(scores1 * scores2) * 1000)
        loss = loss + (
            torch.sum(scores1 * scores2 * F.relu(margin + diff))
            / (torch.sum(scores1 * scores2) + 1e-6)
        )

        has_grad = True
        n_valid_samples += 1

        if plot and batch["batch_idx"] % batch["log_interval"] == 0:
            pos1_aux = pos1.cpu().numpy()
            pos2_aux = pos2.cpu().numpy()
            k = pos1_aux.shape[1]
            col = np.random.rand(k, 3)
            n_sp = 4
            plt.figure()
            plt.subplot(1, n_sp, 1)
            im1 = imshow_image(
                batch["image1"][idx_in_batch].cpu().numpy(),
                preprocessing=batch["preprocessing"],
            )
            plt.imshow(im1)
            plt.scatter(
                pos1_aux[1, :],
                pos1_aux[0, :],
                s=0.25**2,
                c=col,
                marker=",",
                alpha=0.5,
            )
            plt.axis("off")
            plt.subplot(1, n_sp, 2)
            plt.imshow(output["scores1"][idx_in_batch].data.cpu().numpy(), cmap="Reds")
            plt.axis("off")
            plt.subplot(1, n_sp, 3)
            im2 = imshow_image(
                batch["image2"][idx_in_batch].cpu().numpy(),
                preprocessing=batch["preprocessing"],
            )
            plt.imshow(im2)
            plt.scatter(
                pos2_aux[1, :],
                pos2_aux[0, :],
                s=0.25**2,
                c=col,
                marker=",",
                alpha=0.5,
            )
            plt.axis("off")
            plt.subplot(1, n_sp, 4)
            plt.imshow(output["scores2"][idx_in_batch].data.cpu().numpy(), cmap="Reds")
            plt.axis("off")
            savefig(
                "train_vis/%s.%02d.%02d.%d.png"
                % (
                    "train" if batch["train"] else "valid",
                    batch["epoch_idx"],
                    batch["batch_idx"] // batch["log_interval"],
                    idx_in_batch,
                ),
                dpi=300,
            )
            plt.close()

    if not has_grad:
        raise NoGradientError

    loss = loss / n_valid_samples

    return loss


def loss_function_qxs(
    model, batch, device, margin=1, safe_radius=4, scaling_steps=3, plot=False
):
    output = model(
        {"image1": batch["image1"].to(device), "image2": batch["image2"].to(device)}
    )

    loss = torch.tensor(np.array([0], dtype=np.float32), device=device)
    has_grad = False

    n_valid_samples = 0
    for idx_in_batch in range(batch["image1"].size(0)):
        # Annotations
        # depth1 = batch['depth1'][idx_in_batch].to(device)  # [h1, w1]
        # intrinsics1 = batch['intrinsics1'][idx_in_batch].to(device)  # [3, 3]
        # pose1 = batch['pose1'][idx_in_batch].view(4, 4).to(device)  # [4, 4]
        # bbox1 = batch['bbox1'][idx_in_batch].to(device)  # [2]

        # depth2 = batch['depth2'][idx_in_batch].to(device)
        # intrinsics2 = batch['intrinsics2'][idx_in_batch].to(device)
        # pose2 = batch['pose2'][idx_in_batch].view(4, 4).to(device)
        # bbox2 = batch['bbox2'][idx_in_batch].to(device)

        # Network output
        dense_features1 = output["dense_features1"][idx_in_batch]
        c, h1, w1 = dense_features1.size()  # 512，32，32
        scores1 = output["scores1"][idx_in_batch].view(-1)  # 1024

        dense_features2 = output["dense_features2"][idx_in_batch]
        _, h2, w2 = dense_features2.size()
        scores2 = output["scores2"][idx_in_batch]

        all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)
        descriptors1 = all_descriptors1  # [512, 1024=32*32]

        all_descriptors2 = F.normalize(dense_features2.view(c, -1), dim=0)
        descriptors2 = all_descriptors2  # [512, 1024=32*32]

        # Warp the positions from image 1 to image 2
        fmap_pos1 = grid_positions(h1, w1, device)
        pos1 = upscale_positions(fmap_pos1, scaling_steps=scaling_steps)  # [2, 1024]
        try:
            pos1, pos2, ids = warp(
                pos1,
                depth1,
                intrinsics1,
                pose1,
                bbox1,
                depth2,
                intrinsics2,
                pose2,
                bbox2,
            )
        except EmptyTensorError:
            continue
        fmap_pos1 = fmap_pos1[:, ids]
        descriptors1 = descriptors1[:, ids]
        scores1 = scores1[ids]

        # Skip the pair if not enough GT correspondences are available
        if ids.size(0) < 128:
            continue

        # Descriptors at the corresponding positions
        fmap_pos2 = torch.round(
            downscale_positions(pos2, scaling_steps=scaling_steps)
        ).long()
        descriptors2 = F.normalize(
            dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]], dim=0
        )

        positive_distance = (
            2
            - 2
            * (descriptors1.t().unsqueeze(1) @ descriptors2.t().unsqueeze(2)).squeeze()
        )

        all_fmap_pos2 = grid_positions(h2, w2, device)
        position_distance = torch.max(
            torch.abs(fmap_pos2.unsqueeze(2).float() - all_fmap_pos2.unsqueeze(1)),
            dim=0,
        )[0]
        is_out_of_safe_radius = position_distance > safe_radius
        distance_matrix = 2 - 2 * (descriptors1.t() @ all_descriptors2)
        negative_distance2 = torch.min(
            distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.0, dim=1
        )[0]

        all_fmap_pos1 = grid_positions(h1, w1, device)
        position_distance = torch.max(
            torch.abs(fmap_pos1.unsqueeze(2).float() - all_fmap_pos1.unsqueeze(1)),
            dim=0,
        )[0]
        is_out_of_safe_radius = position_distance > safe_radius
        distance_matrix = 2 - 2 * (descriptors2.t() @ all_descriptors1)
        negative_distance1 = torch.min(
            distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.0, dim=1
        )[0]

        diff = positive_distance - torch.min(negative_distance1, negative_distance2)

        scores2 = scores2[fmap_pos2[0, :], fmap_pos2[1, :]]

        loss = loss + (
            torch.sum(scores1 * scores2 * F.relu(margin + diff))
            / torch.sum(scores1 * scores2)
        )

        has_grad = True
        n_valid_samples += 1

        if plot and batch["batch_idx"] % batch["log_interval"] == 0:
            pos1_aux = pos1.cpu().numpy()
            pos2_aux = pos2.cpu().numpy()
            k = pos1_aux.shape[1]
            col = np.random.rand(k, 3)
            n_sp = 4
            plt.figure()
            plt.subplot(1, n_sp, 1)
            im1 = imshow_image(
                batch["image1"][idx_in_batch].cpu().numpy(),
                preprocessing=batch["preprocessing"],
            )
            plt.imshow(im1)
            plt.scatter(
                pos1_aux[1, :],
                pos1_aux[0, :],
                s=0.25**2,
                c=col,
                marker=",",
                alpha=0.5,
            )
            plt.axis("off")
            plt.subplot(1, n_sp, 2)
            plt.imshow(output["scores1"][idx_in_batch].data.cpu().numpy(), cmap="Reds")
            plt.axis("off")
            plt.subplot(1, n_sp, 3)
            im2 = imshow_image(
                batch["image2"][idx_in_batch].cpu().numpy(),
                preprocessing=batch["preprocessing"],
            )
            plt.imshow(im2)
            plt.scatter(
                pos2_aux[1, :],
                pos2_aux[0, :],
                s=0.25**2,
                c=col,
                marker=",",
                alpha=0.5,
            )
            plt.axis("off")
            plt.subplot(1, n_sp, 4)
            plt.imshow(output["scores2"][idx_in_batch].data.cpu().numpy(), cmap="Reds")
            plt.axis("off")
            savefig(
                "train_vis/%s.%02d.%02d.%d.png"
                % (
                    "train" if batch["train"] else "valid",
                    batch["epoch_idx"],
                    batch["batch_idx"] // batch["log_interval"],
                    idx_in_batch,
                ),
                dpi=300,
            )
            plt.close()

    if not has_grad:
        raise NoGradientError

    loss = loss / n_valid_samples

    return loss


def interpolate_depth(pos, depth):
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)

    h, w = depth.size()

    i = pos[0, :]
    j = pos[1, :]

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right),
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # Valid depth
    valid_depth = torch.min(
        torch.min(
            depth[i_top_left, j_top_left] > 0, depth[i_top_right, j_top_right] > 0
        ),
        torch.min(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0,
        ),
    )

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    ids = ids[valid_depth]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left]
        + w_top_right * depth[i_top_right, j_top_right]
        + w_bottom_left * depth[i_bottom_left, j_bottom_left]
        + w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    return [interpolated_depth, pos, ids]


def uv_to_pos(uv):
    return torch.cat([uv[1, :].view(1, -1), uv[0, :].view(1, -1)], dim=0)


def warp(pos1, depth1, intrinsics1, pose1, bbox1, depth2, intrinsics2, pose2, bbox2):
    device = pos1.device

    Z1, pos1, ids = interpolate_depth(pos1, depth1)

    # COLMAP convention
    u1 = pos1[1, :] + bbox1[1] + 0.5
    v1 = pos1[0, :] + bbox1[0] + 0.5

    X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
    Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])

    XYZ1_hom = torch.cat(
        [
            X1.view(1, -1),
            Y1.view(1, -1),
            Z1.view(1, -1),
            torch.ones(1, Z1.size(0), device=device),
        ],
        dim=0,
    )
    # XYZ2_hom = torch.chain_matmul(pose2, torch.inverse(pose1), XYZ1_hom)
    XYZ2_hom = torch.linalg.multi_dot((pose2, torch.inverse(pose1), XYZ1_hom))
    XYZ2 = XYZ2_hom[:-1, :] / XYZ2_hom[-1, :].view(1, -1)

    # uv2_hom = torch.matmul(intrinsics2, XYZ2)
    uv2_hom = torch.matmul(intrinsics2, XYZ2)
    uv2 = uv2_hom[:-1, :] / uv2_hom[-1, :].view(1, -1)

    u2 = uv2[0, :] - bbox2[1] - 0.5
    v2 = uv2[1, :] - bbox2[0] - 0.5
    uv2 = torch.cat([u2.view(1, -1), v2.view(1, -1)], dim=0)

    annotated_depth, pos2, new_ids = interpolate_depth(uv_to_pos(uv2), depth2)

    ids = ids[new_ids]
    pos1 = pos1[:, new_ids]
    estimated_depth = XYZ2[2, new_ids]

    inlier_mask = torch.abs(estimated_depth - annotated_depth) < 0.05

    ids = ids[inlier_mask]
    if ids.size(0) == 0:
        raise EmptyTensorError

    pos2 = pos2[:, inlier_mask]
    pos1 = pos1[:, inlier_mask]

    return pos1, pos2, ids


def loss_l2(model, batch, device):
    # model.to(device)
    output0 = model(batch["image1"].to(device))
    output1 = model(batch["image2"].to(device))

    dense_feat_map0 = output0["dense_feat_map"]
    dense_feat_map1 = output1["dense_feat_map"]
    pos0 = output0["kpts"]
    pos1 = output1["kpts"]
    score_map0 = output0["score_map"]
    score_map1 = output1["score_map"]

    batch_size = pos0.shape[0]
    num_corr = pos0.shape[1]
    loss_type = "L2NET"
    config = None

    loss, acc = make_detector_loss(
        pos0,
        pos1,
        dense_feat_map0,
        dense_feat_map1,
        score_map0,
        score_map1,
        batch_size,
        num_corr,
        loss_type,
        config,
    )
    return loss


def make_detector_loss(
    pos0,
    pos1,
    dense_feat_map0,
    dense_feat_map1,
    score_map0,
    score_map1,
    batch_size,
    num_corr,
    loss_type,
    config,
):
    joint_loss = torch.tensor(0.0)
    accuracy = torch.tensor(0.0)
    all_valid_pos0 = []
    all_valid_pos1 = []
    all_valid_match = []
    for i in range(batch_size):
        # random sample
        valid_pos0, valid_pos1 = rnd_sample([pos0[i], pos1[i]], num_corr)
        valid_pos0 = valid_pos0.unsqueeze(0)
        valid_pos1 = valid_pos1.unsqueeze(0)
        valid_num = valid_pos0.shape[1]

        valid_feat0 = interpolate(valid_pos0 / 4, dense_feat_map0[i], batched=True)
        valid_feat1 = interpolate(valid_pos1 / 4, dense_feat_map1[i], batched=True)

        valid_feat0 = F.normalize(valid_feat0, dim=-1)
        valid_feat1 = F.normalize(valid_feat1, dim=-1)

        valid_score0 = interpolate(
            valid_pos0, score_map0[i].squeeze(dim=-1), batched=True
        )
        valid_score1 = interpolate(
            valid_pos1, score_map1[i].squeeze(dim=-1), batched=True
        )
        # if config["det"]["corr_weight"]:
        if True:
            corr_weight = valid_score0 * valid_score1
        else:
            corr_weight = None

        # safe_radius = config["det"]["safe_radius"]
        safe_radius = 3

        if safe_radius > 0:
            radius_mask_row = get_dist_mat(
                valid_pos1, valid_pos1, "euclidean_dist_no_norm"
            )
            radius_mask_row = radius_mask_row < safe_radius

            radius_mask_col = get_dist_mat(
                valid_pos0, valid_pos0, "euclidean_dist_no_norm"
            )
            radius_mask_col = radius_mask_col < safe_radius

            radius_mask_row = radius_mask_row.float() - torch.eye(valid_num).cuda()
            radius_mask_col = radius_mask_col.float() - torch.eye(valid_num).cuda()
        else:
            radius_mask_row = None
            radius_mask_col = None

        # 有效的匹配点的数量过少
        if valid_num < 32:
            si_loss = torch.tensor(0.0)
            si_accuracy = torch.tensor(1.0)
            matched_mask = torch.zeros((1, valid_num), dtype=torch.bool)
        else:
            si_loss, si_accuracy, matched_mask = make_structured_loss(
                valid_feat0,
                valid_feat1,
                loss_type=loss_type,
                radius_mask_row=radius_mask_row,
                radius_mask_col=radius_mask_col,
                corr_weight=corr_weight if corr_weight is not None else None,
                name="si_loss",
            )

        joint_loss += si_loss.cpu() / batch_size
        accuracy += si_accuracy.cpu() / batch_size
        all_valid_match.append(matched_mask.squeeze(dim=0))
        all_valid_pos0.append(valid_pos0)
        all_valid_pos1.append(valid_pos1)

    return joint_loss, accuracy


def make_quadruple_loss(kpt_m0, kpt_m1, inlier_num):
    batch_size = kpt_m0.size(0)
    num_corr = kpt_m1.size(1)
    kpt_m_diff0 = torch.transpose(kpt_m0.repeat(1, 1, num_corr), 1, 2) - kpt_m0
    kpt_m_diff1 = torch.transpose(kpt_m1.repeat(1, 1, num_corr), 1, 2) - kpt_m1

    R = kpt_m_diff0 * kpt_m_diff1

    quad_loss = 0
    accuracy = 0
    for i in range(batch_size):
        cur_inlier_num = inlier_num[i].squeeze()
        inlier_block = R[i, 0:cur_inlier_num, 0:cur_inlier_num]
        inlier_block = inlier_block + torch.eye(cur_inlier_num)
        inlier_block = torch.maximum(torch.tensor(0.0), 1.0 - inlier_block)
        error = torch.count_nonzero(inlier_block)
        cur_inlier_num = cur_inlier_num.to(torch.float32)
        quad_loss += torch.sum(inlier_block) / (cur_inlier_num * (cur_inlier_num - 1))
        accuracy += 1.0 - error.to(torch.float32) / (
            cur_inlier_num * (cur_inlier_num - 1)
        )

    quad_loss /= float(batch_size)
    accuracy /= float(batch_size)
    return quad_loss, accuracy


def make_structured_loss(
    feat_anc,
    feat_pos,
    loss_type="RATIO",
    inlier_mask=None,
    radius_mask_row=None,
    radius_mask_col=None,
    corr_weight=None,
    dist_mat=None,
    name="loss",
):
    batch_size = feat_anc.size(0)
    num_corr = feat_anc.size(1)
    if inlier_mask is None:
        inlier_mask = torch.ones((batch_size, num_corr), dtype=torch.bool)
    inlier_num = torch.count_nonzero(inlier_mask, dim=-1)

    if loss_type == "LOG" or loss_type == "L2NET" or loss_type == "CIRCLE":
        dist_type = "cosine_dist"
    elif loss_type.find("HARD") >= 0:
        dist_type = "euclidean_dist"
    else:
        raise NotImplementedError()

    if dist_mat is None:
        dist_mat = get_dist_mat(feat_anc, feat_pos, dist_type)
    pos_vec = torch.diagonal(dist_mat, dim1=-2, dim2=-1)

    if loss_type.find("HARD") >= 0:
        neg_margin = 1
        dist_mat_without_min_on_diag = dist_mat + 10 * torch.eye(num_corr).unsqueeze(0)

        mask = torch.less(dist_mat_without_min_on_diag, 0.008).float()
        dist_mat_without_min_on_diag += mask * 10

        if radius_mask_row is not None:
            hard_neg_dist_row = dist_mat_without_min_on_diag + 10 * radius_mask_row
        else:
            hard_neg_dist_row = dist_mat_without_min_on_diag
        if radius_mask_col is not None:
            hard_neg_dist_col = dist_mat_without_min_on_diag + 10 * radius_mask_col
        else:
            hard_neg_dist_col = dist_mat_without_min_on_diag

        hard_neg_dist_row = torch.min(hard_neg_dist_row, dim=-1)[0]
        hard_neg_dist_col = torch.min(hard_neg_dist_col, dim=-2)[0]

        if loss_type == "HARD_TRIPLET":
            loss_row = torch.maximum(neg_margin + pos_vec - hard_neg_dist_row, 0)
            loss_col = torch.maximum(neg_margin + pos_vec - hard_neg_dist_col, 0)
        elif loss_type == "HARD_CONTRASTIVE":
            pos_margin = 0.2
            pos_loss = torch.maximum(pos_vec - pos_margin, 0)
            loss_row = pos_loss + torch.maximum(neg_margin - hard_neg_dist_row, 0)
            loss_col = pos_loss + torch.maximum(neg_margin - hard_neg_dist_col, 0)
        else:
            raise NotImplementedError()

    elif loss_type == "LOG" or loss_type == "L2NET":
        if loss_type == "LOG":
            log_scale = torch.nn.Parameter(torch.tensor(1.0))
            torch.nn.init.constant_(log_scale, 1.0)
        else:
            log_scale = torch.tensor(1.0)

        softmax_row = F.softmax(log_scale * dist_mat, dim=1)
        softmax_col = F.softmax(log_scale * dist_mat, dim=0)

        loss_row = -torch.log(torch.diagonal(softmax_row, dim1=-2, dim2=-1))
        loss_col = -torch.log(torch.diagonal(softmax_col, dim1=-2, dim2=-1))

    elif loss_type == "CIRCLE":
        log_scale = 512
        m = 0.1
        neg_mask_row = torch.eye(num_corr).unsqueeze(0)
        if radius_mask_row is not None:
            neg_mask_row += radius_mask_row
        neg_mask_col = torch.eye(num_corr).unsqueeze(0)
        if radius_mask_col is not None:
            neg_mask_col += radius_mask_col

        pos_margin = 1 - m
        neg_margin = m
        pos_optimal = 1 + m
        neg_optimal = -m

        neg_mat_row = dist_mat - 128 * neg_mask_row
        neg_mat_col = dist_mat - 128 * neg_mask_col

        lse_positive = torch.logsumexp(
            -log_scale
            * (pos_vec.unsqueeze(-1) - pos_margin)
            * torch.maximum(pos_optimal - pos_vec.unsqueeze(-1), torch.tensor(0)),
            dim=-1,
        )

        lse_negative_row = torch.logsumexp(
            log_scale
            * (neg_mat_row - neg_margin)
            * torch.maximum(neg_mat_row - neg_optimal, torch.tensor(0)),
            dim=-1,
        )

        lse_negative_col = torch.logsumexp(
            log_scale
            * (neg_mat_col - neg_margin)
            * torch.maximum(neg_mat_col - neg_optimal, torch.tensor(0)),
            dim=-2,
        )

        loss_row = F.softplus(lse_positive + lse_negative_row) / log_scale
        loss_col = F.softplus(lse_positive + lse_negative_col) / log_scale

    else:
        raise NotImplementedError()

    if dist_type == "cosine_dist":
        err_row = dist_mat - pos_vec.unsqueeze(-1)
        err_col = dist_mat - pos_vec.unsqueeze(-2)
    elif dist_type == "euclidean_dist" or dist_type == "euclidean_dist_no_norm":
        err_row = pos_vec.unsqueeze(-1) - dist_mat
        err_col = pos_vec.unsqueeze(-2) - dist_mat
    else:
        raise NotImplementedError()
    if radius_mask_row is not None:
        err_row = err_row - 10 * radius_mask_row
    if radius_mask_col is not None:
        err_col = err_col - 10 * radius_mask_col
    err_row = torch.sum(torch.maximum(err_row, torch.tensor(0.0)), dim=-1)
    err_col = torch.sum(torch.maximum(err_col, torch.tensor(0.0)), dim=-2)

    loss = 0
    accuracy = 0

    tot_loss = (loss_row + loss_col) / 2
    if corr_weight is not None:
        tot_loss = tot_loss * corr_weight

    for i in range(batch_size):
        if corr_weight is not None:
            loss += torch.sum(tot_loss[i][inlier_mask[i]]) / (
                torch.sum(corr_weight[i][inlier_mask[i]]) + 1e-6
            )
        else:
            loss += torch.mean(tot_loss[i][inlier_mask[i]])

        err_row = err_row.unsqueeze(0)
        err_col = err_col.unsqueeze(0)

        cnt_err_row = torch.count_nonzero(err_row[i][inlier_mask[i]])
        cnt_err_col = torch.count_nonzero(err_col[i][inlier_mask[i]])

        tot_err = cnt_err_row + cnt_err_col

        accuracy += 1.0 - torch.div(tot_err, inlier_num[i].to(torch.float32)) / (
            batch_size * 2.0
        )

    # matched_mask = torch.logical_and(
    #     torch.equal(err_row, torch.tensor(0.0)), torch.equal(err_col, torch.tensor(0.0))
    # )
    matched_mask = (err_row == 0.0) & (err_col == 0.0)

    matched_mask = torch.logical_and(matched_mask.cuda(), inlier_mask.cuda())

    loss /= batch_size
    accuracy /= batch_size

    return loss, accuracy, matched_mask
