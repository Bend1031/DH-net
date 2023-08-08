import torch
import torch.nn.functional as F


def rnd_sample(inputs, n_sample, seed=1895):
    cur_size = inputs[0].shape[0]
    rnd_idx = torch.randperm(cur_size, generator=torch.Generator().manual_seed(seed))[
        :n_sample
    ]
    outputs = [i[rnd_idx] for i in inputs]
    return outputs


def get_dist_mat(feat1, feat2, dist_type):
    eps = torch.Tensor([1e-6]).cuda()
    # eps = [1e-6]
    feat1.squeeze_(dim=0)
    feat2.squeeze_(dim=0)
    cos_dist_mat = torch.matmul(feat1, feat2.t())

    if dist_type == "cosine_dist":
        dist_mat = torch.clamp(cos_dist_mat, -1, 1)
    elif dist_type == "euclidean_dist":
        dist_mat = torch.sqrt(torch.maximum(2 - 2 * cos_dist_mat, eps))
    elif dist_type == "euclidean_dist_no_norm":
        norm1 = torch.sum(feat1 * feat1, dim=-1, keepdim=True)
        norm2 = torch.sum(feat2 * feat2, dim=-1, keepdim=True)
        dist_mat = torch.sqrt(
            torch.maximum(
                torch.Tensor([0.0]).cuda(), norm1 - 2 * cos_dist_mat + norm2.t()
            )
            + eps
        )
    else:
        raise NotImplementedError()
    return dist_mat


# def interpolate(pos, inputs, batched=True, nd=True):
#     if not batched:
#         pos = pos.unsqueeze(0)
#         inputs = inputs.unsqueeze(0)

#     h = inputs.shape[1]
#     w = inputs.shape[2]

#     i = pos[:, :, 0]
#     j = pos[:, :, 1]

#     i_top_left = torch.clamp(torch.floor(i).to(torch.int32), 0, h - 1)
#     j_top_left = torch.clamp(torch.floor(j).to(torch.int32), 0, w - 1)

#     i_top_right = torch.clamp(torch.floor(i).to(torch.int32), 0, h - 1)
#     j_top_right = torch.clamp(torch.ceil(j).to(torch.int32), 0, w - 1)

#     i_bottom_left = torch.clamp(torch.ceil(i).to(torch.int32), 0, h - 1)
#     j_bottom_left = torch.clamp(torch.floor(j).to(torch.int32), 0, w - 1)

#     i_bottom_right = torch.clamp(torch.ceil(i).to(torch.int32), 0, h - 1)
#     j_bottom_right = torch.clamp(torch.ceil(j).to(torch.int32), 0, w - 1)

#     dist_i_top_left = i - i_top_left.float()
#     dist_j_top_left = j - j_top_left.float()
#     w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
#     w_top_right = (1 - dist_i_top_left) * dist_j_top_left
#     w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
#     w_bottom_right = dist_i_top_left * dist_j_top_left

#     if nd:
#         w_top_left = w_top_left[..., None]
#         w_top_right = w_top_right[..., None]
#         w_bottom_left = w_bottom_left[..., None]
#         w_bottom_right = w_bottom_right[..., None]

#     interpolated_val = (
#         w_top_left
#         * inputs.gather(
#             dim=1, index=torch.stack([i_top_left, j_top_left], dim=-1).long()
#         )
#         + w_top_right
#         * inputs.gather(
#             dim=1, index=torch.stack([i_top_right, j_top_right], dim=-1).long()
#         )
#         + w_bottom_left
#         * inputs.gather(
#             dim=1, index=torch.stack([i_bottom_left, j_bottom_left], dim=-1).long()
#         )
#         + w_bottom_right
#         * inputs.gather(
#             dim=1, index=torch.stack([i_bottom_right, j_bottom_right], dim=-1).long()
#         )
#     )

#     if not batched:
#         interpolated_val = interpolated_val.squeeze(dim=0)
#     return interpolated_val


def interpolate(pos, inputs, batched=True, nd=True):
    """
    Perform bilinear interpolation on inputs based on pos coordinates using F.grid_sample.

    Args:
    - inputs (torch.Tensor): Input tensor of shape (batchsize, channels, height, width).
    - pos (torch.Tensor): Position tensor of shape (batchsize, point_nums, 2) containing x and y coordinates.

    Returns:
    - output (torch.Tensor): Interpolated output tensor of shape (batchsize, point_nums, channels).

    img.shape : [B,C,H_in,W_in]
    grid.shape: [B,H_out,W_out,2]
    out: [B,C,H_out,W_out]
    """
    if inputs.dim() != 4:
        inputs = inputs.unsqueeze(0)
    batchsize, channels, height, width = inputs.shape
    point_nums = pos.shape[1]

    # 归一化位置坐标到[-1, 1]范围，以适应grid_sample函数的要求
    pos_normalized = pos.clone()
    pos_normalized[:, :, 0] = 2 * pos[:, :, 0] / (width - 1) - 1
    pos_normalized[:, :, 1] = 2 * pos[:, :, 1] / (height - 1) - 1

    # 将位置坐标转为网格，为了使用grid_sample函数
    grid = pos_normalized.view(batchsize, point_nums, 1, 2)
    # shape: [1, point_nums, 1, 2]

    # 使用grid_sample进行插值采样
    sampled_output = F.grid_sample(inputs, grid, align_corners=True)

    # 调整输出形状为[1, point_nums, channels]
    sampled_output = sampled_output.view(batchsize, point_nums, channels)

    # print(sampled_output.shape)  # 输出: torch.Size([1, point_nums, channels])

    return sampled_output
