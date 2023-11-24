import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Net(nn.Module):
    def __init__(self, is_muldet=True, device="cuda"):
        super(L2Net, self).__init__()
        self.is_muldet = is_muldet

        self.conv0 = conv(3, 32, kernel_size=3, stride=1, dilation=1)
        self.conv1 = conv(32, 32, kernel_size=3, stride=1, dilation=1)

        self.conv2 = conv(32, 64, kernel_size=3, stride=2, dilation=1)
        self.conv3 = conv(64, 64, kernel_size=3, stride=1, dilation=1)

        self.conv4 = conv(64, 128, kernel_size=3, stride=2, dilation=1)
        self.conv5 = conv(128, 128, kernel_size=3, stride=1, dilation=1)
        self.conv6 = conv(128, 128, kernel_size=3, stride=1, dilation=1)
        self.conv7 = conv(128, 128, kernel_size=3, stride=1, dilation=1)

        self.detection1 = MulDet(level=1)
        self.detection2 = MulDet(level=2)
        self.detection4 = MulDet(level=4)

    def extract_kpts(
        self, score_map, k=4000, score_thld=0.0, edge_thld=0, nms_size=3, eof_size=5
    ):
        """
        Extracts keypoints from a score map.
        score_map: 输入的得分图，它是一个4维的张量，形状为 (batch_size, num_channels, height, width)，表示了每个像素点对应的得分。

        k: 需要提取的关键点数量。默认为 4000。函数会根据这个参数选择得分最高的 k 个关键点。

        score_thld: 得分的阈值。默认为 0.0。低于这个阈值的像素点会被忽略。

        edge_thld: 边缘阈值。默认为 0。这个参数似乎会传递给一个名为 edge_mask 的方法，用于生成一个边缘掩码，但是在这段代码中，edge_mask 的实现并没有被提供。

        nms_size: 非极大值抑制 (NMS) 的窗口大小。默认为 3。NMS 用于抑制重叠的关键点，保留得分最高的关键点。

        eof_size: 边缘截断的大小。默认为 5。这个参数用于在得分图的边缘位置进行截断，以去除边缘可能引入的噪声。
        """
        h, w = score_map.shape[2], score_map.shape[3]

        mask = score_map > score_thld
        if nms_size > 0:
            nms_mask = F.max_pool2d(
                score_map, kernel_size=nms_size, stride=1, padding=nms_size // 2
            )
            nms_mask = score_map == nms_mask
            mask = nms_mask & mask
        if eof_size > 0:
            eof_mask = torch.ones(
                (1, 1, h - 2 * eof_size, w - 2 * eof_size),
                dtype=torch.bool,
                device="cuda",
            )
            eof_mask = F.pad(eof_mask, (eof_size, eof_size, eof_size, eof_size))
            mask = eof_mask & mask
        if edge_thld > 0:
            edge_mask = self.edge_mask(
                score_map, 1, dilation=3, edge_thld=edge_thld
            )  # Assuming self.edge_mask is implemented
            mask = edge_mask & mask

        mask = mask.view(h, w)
        score_map = score_map.view(h, w)
        indices = torch.nonzero(mask, as_tuple=False)
        scores = score_map[indices[:, 0], indices[:, 1]]
        if k > indices.shape[0]:
            k = indices.shape[0]
        top_scores, top_indices = torch.topk(scores, k, largest=True)
        indices = indices[top_indices].unsqueeze(0)
        scores = top_scores.unsqueeze(0)

        return indices, scores

    def edge_mask(self, inputs, n_channel, dilation=1, edge_thld=5):
        """
        标记非边缘区域
        """
        dii_filter = torch.tensor(
            [[0, 1.0, 0], [0, -2.0, 0], [0, 1.0, 0]], dtype=torch.float32, device="cuda"
        )
        dij_filter = 0.25 * torch.tensor(
            [[1.0, 0, -1.0], [0, 0.0, 0], [-1.0, 0, 1.0]],
            dtype=torch.float32,
            device="cuda",
        )
        djj_filter = torch.tensor(
            [[0, 0, 0], [1.0, -2.0, 1.0], [0, 0, 0]], dtype=torch.float32, device="cuda"
        )

        dii_filter = dii_filter.view(1, 1, 3, 3).repeat(1, n_channel, 1, 1)
        dij_filter = dij_filter.view(1, 1, 3, 3).repeat(1, n_channel, 1, 1)
        djj_filter = djj_filter.view(1, 1, 3, 3).repeat(1, n_channel, 1, 1)

        pad_inputs = F.pad(inputs, (dilation, dilation, dilation, dilation))

        dii = F.conv2d(pad_inputs, dii_filter, stride=1, padding=0, dilation=dilation)
        dij = F.conv2d(pad_inputs, dij_filter, stride=1, padding=0, dilation=dilation)
        djj = F.conv2d(pad_inputs, djj_filter, stride=1, padding=0, dilation=dilation)

        det = dii * djj - dij * dij
        tr = dii + djj
        thld = (edge_thld + 1) ** 2 / edge_thld
        is_not_edge = (tr * tr / det <= thld) & (det > 0)

        return is_not_edge

    def kpt_refinement(self, inputs):
        """关键点精细化"""
        n_channel = inputs.size(1)

        di_filter = torch.tensor(
            [[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], dtype=torch.float32, device="cuda"
        ).view(1, 1, 3, 3)
        dj_filter = torch.tensor(
            [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], dtype=torch.float32, device="cuda"
        ).view(1, 1, 3, 3)
        dii_filter = torch.tensor(
            [[0, 1.0, 0], [0, -2.0, 0], [0, 1.0, 0]], dtype=torch.float32, device="cuda"
        ).view(1, 1, 3, 3)
        dij_filter = (
            torch.tensor(
                [[1.0, 0, -1.0], [0, 0.0, 0], [-1.0, 0, 1.0]],
                dtype=torch.float32,
                device="cuda",
            )
            .mul(0.25)
            .view(1, 1, 3, 3)
        )
        djj_filter = torch.tensor(
            [[0, 0, 0], [1.0, -2.0, 1.0], [0, 0, 0]], dtype=torch.float32, device="cuda"
        ).view(1, 1, 3, 3)

        dii_filter = dii_filter.repeat(1, n_channel, 1, 1)
        dii = F.conv2d(inputs, dii_filter, padding=1)

        dij_filter = dij_filter.repeat(1, n_channel, 1, 1)
        dij = F.conv2d(inputs, dij_filter, padding=1)

        djj_filter = djj_filter.repeat(1, n_channel, 1, 1)
        djj = F.conv2d(inputs, djj_filter, padding=1)

        det = dii * djj - dij * dij

        inv_hess_00 = djj / det
        inv_hess_01 = -dij / det
        inv_hess_11 = dii / det

        di_filter = di_filter.repeat(1, n_channel, 1, 1)
        di = F.conv2d(inputs, di_filter, padding=1)

        dj_filter = dj_filter.repeat(1, n_channel, 1, 1)
        dj = F.conv2d(inputs, dj_filter, padding=1)

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)

        return torch.stack([step_i, step_j], dim=-1)

    def forward(self, x):
        x.to("cuda")
        self.ori_h = x.shape[2]
        self.ori_w = x.shape[3]

        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)

        # L2 norm
        dense_feat_map = conv7  # (1, 128, 128, 128)
        # dense_feat_map = F.normalize(conv7, p=2, dim=2)

        if self.is_muldet:
            score_map1 = self.detection1(conv1)
            score_map2 = self.detection2(conv3)
            score_map4 = self.detection4(conv7)
            # 根据level加权求和
            score_map = (1 * score_map1 + 2 * score_map2 + 3 * score_map4) / 6
        else:
            score_map = self.detection4(conv7)

        # (1,4000,2) (1,4000)
        kpt_inds, kpt_score = self.extract_kpts(
            score_map,
            k=1000,
            score_thld=0.0,
            edge_thld=10,
            nms_size=3,
            eof_size=5,
        )

        # if det_config["kpt_refinement"]:
        if True:
            offsets = torch.squeeze(self.kpt_refinement(score_map), dim=1).view(
                1, -1, 2
            )
            offsets = offsets.gather(1, kpt_inds)
            offsets = torch.clamp(offsets, -0.5, 0.5)
            kpt_inds = kpt_inds.float() + offsets
        else:
            kpt_inds = kpt_inds.float()

        # 坐标索引需要缩放四倍，从而与特征图大小对应
        # p=2 L2 norm
        # dim=-1

        # 特征描述符 [1,,4000,128]
        descs = interpolate(kpt_inds / 4, dense_feat_map)
        descs = F.normalize(descs, p=2, dim=-1)

        # descs = descs.permute(0, 2, 1, 3).squeeze(-1)
        # 坐标点位置 [n,2]
        kpts = torch.stack([kpt_inds[:, :, 1], kpt_inds[:, :, 0]], dim=-1)

        # 坐标点置信度分数 [n]
        kpt_score = kpt_score.clone().detach()

        return {
            "dense_feat_map": dense_feat_map,  # [1,128,128,128]
            "descs": descs,  # [1,4000,128]
            "kpts": kpts,  # [1,4000,2]
            "kpt_score": kpt_score,  # [1,4000]
            "score_map": score_map,  # [1,1,512,512]
        }
        # "displacements": displacements,


class MulDet(nn.Module):
    def __init__(self, level=1, ori_h=512, ori_w=512):
        super(MulDet, self).__init__()
        self.level = level
        self.ori_h = ori_h
        self.ori_w = ori_w

    def peakiness_score(self, inputs, ksize=3, need_norm=True, dilation=1):
        if need_norm:
            with torch.inference_mode():
                instance_max = torch.max(inputs)
                inputs = inputs / instance_max

        pad_inputs = F.pad(
            inputs, (dilation, dilation, dilation, dilation), mode="reflect"
        )
        avg_inputs = F.avg_pool2d(pad_inputs, kernel_size=ksize, stride=1)
        alpha = F.softplus(inputs - avg_inputs)
        beta = F.softplus(inputs - torch.mean(inputs, dim=-1, keepdim=True))
        return alpha, beta

    # def upsample(self, feature_map, level):
    #     pass

    def forward(self, x):
        alpha, beta = self.peakiness_score(x)
        score_vol = alpha * beta  # ([1, 32, 512, 512])，
        score_map = torch.max(score_vol, dim=1, keepdim=True)[0]

        score_map = F.interpolate(
            score_map, size=(self.ori_h, self.ori_w), mode="bilinear"
        )

        return score_map


# def interpolate(pos, inputs, batched=True, nd=True):
#     if not batched:
#         pos = pos.unsqueeze(0)
#         inputs = inputs.unsqueeze(0)

#     b, c, h, w = inputs.shape  # 获取输入图像的通道数和高度、宽度

#     n = pos.shape[1]  # 坐标点数量
#     i = pos[:, :, 0]
#     j = pos[:, :, 1]

#     # 计算四个相邻位置的索引
#     i_top_left = torch.clamp(torch.floor(i).to(torch.int64), 0, h - 1)
#     j_top_left = torch.clamp(torch.floor(j).to(torch.int64), 0, w - 1)
#     i_top_right = torch.clamp(torch.floor(i).to(torch.int64), 0, h - 1)
#     j_top_right = torch.clamp(torch.ceil(j).to(torch.int64), 0, w - 1)
#     i_bottom_left = torch.clamp(torch.ceil(i).to(torch.int64), 0, h - 1)
#     j_bottom_left = torch.clamp(torch.floor(j).to(torch.int64), 0, w - 1)
#     i_bottom_right = torch.clamp(torch.ceil(i).to(torch.int64), 0, h - 1)
#     j_bottom_right = torch.clamp(torch.ceil(j).to(torch.int64), 0, w - 1)

#     # 计算插值权重
#     dist_i_top_left = i - i_top_left.to(torch.float32)
#     dist_j_top_left = j - j_top_left.to(torch.float32)
#     w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
#     w_top_right = (1 - dist_i_top_left) * dist_j_top_left
#     w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
#     w_bottom_right = dist_i_top_left * dist_j_top_left

#     if nd:
#         # 扩展权重张量的维度
#         w_top_left = w_top_left.unsqueeze(-1)
#         w_top_right = w_top_right.unsqueeze(-1)
#         w_bottom_left = w_bottom_left.unsqueeze(-1)
#         w_bottom_right = w_bottom_right.unsqueeze(-1)

#     # 执行双线性插值
#     interpolated_val = (
#         w_top_left
#         * inputs.gather(2, j_top_left.unsqueeze(1).unsqueeze(-1).expand(-1, c, n, -1))
#         + w_top_right
#         * inputs.gather(2, j_top_right.unsqueeze(1).unsqueeze(-1).expand(-1, c, n, -1))
#         + w_bottom_left
#         * inputs.gather(
#             2, j_bottom_left.unsqueeze(1).unsqueeze(-1).expand(-1, c, n, -1)
#         )
#         + w_bottom_right
#         * inputs.gather(
#             2, j_bottom_right.unsqueeze(1).unsqueeze(-1).expand(-1, c, n, -1)
#         )
#     )

#     if not batched:
#         interpolated_val = interpolated_val.squeeze(0)
#     return interpolated_val

# def interpolate(pos,inputs):
#     batchsize = inputs.shape[0]
#     height = inputs.shape[2]
#     width = inputs.shape[3]
#     y_coords = torch.linspace(-1, 1, height)
#     x_coords = torch.linspace(-1, 1, width)
#     grid_y, grid_x = torch.meshgrid(y_coords, x_coords)
#     grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0)  # 添加batch维度
#     grid = grid.repeat(batchsize, 1, 1, 1, 1)

#     # 转换pos为在grid上的相对坐标
#     grid_pos = (pos / torch.tensor([width, height])) * 2 - 1

#     # 使用grid_sample执行双线性插值
#     outputs = F.grid_sample(inputs, grid + grid_pos.unsqueeze(3), align_corners=True)

#     return outputs


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


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=((kernel_size - 1) // 2) * dilation,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )
