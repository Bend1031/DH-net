import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DenseFeatureExtractionModule(nn.Module):
    def __init__(self, finetune_feature_extraction=False, use_cuda=True):
        super(DenseFeatureExtractionModule, self).__init__()

        # 使用预训练模型
        # model = models.vgg16(pretrained=True)
        model = models.vgg16()
        # print(model)
        vgg16_layers = [
            "conv1_1",
            "relu1_1",
            "conv1_2",
            "relu1_2",
            "pool1",
            "conv2_1",
            "relu2_1",
            "conv2_2",
            "relu2_2",
            "pool2",
            "conv3_1",
            "relu3_1",
            "conv3_2",
            "relu3_2",
            "conv3_3",
            "relu3_3",
            "pool3",
            "conv4_1",
            "relu4_1",
            "conv4_2",
            "relu4_2",
            "conv4_3",
            "relu4_3",
            "pool4",
            "conv5_1",
            "relu5_1",
            "conv5_2",
            "relu5_2",
            "conv5_3",
            "relu5_3",
            "pool5",
        ]
        conv4_3_idx = vgg16_layers.index("conv4_3")

        self.model = nn.Sequential(*list(model.features.children())[: conv4_3_idx + 1])

        self.num_channels = 256

        # Fix forward parameters
        for param in self.model.parameters():
            param.requires_grad = False
        if finetune_feature_extraction:
            # Unlock conv4_3
            for param in list(self.model.parameters())[-2:]:
                param.requires_grad = True

        if use_cuda:
            self.model = self.model.cuda()

    def forward(self, batch):
        output = self.model(batch)
        return output


class SoftDetectionModule(nn.Module):
    """实现了一个软检测（Soft Detection）算法。软检测是一种寻找局部最大值的方法，通过计算输入特征图的每个点相对于其邻域内其他点的响应强度，来生成一个软局部最大值得分图。该方法可以用于关键点检测等任务"""

    def __init__(self, soft_local_max_size=3):
        super(SoftDetectionModule, self).__init__()

        self.soft_local_max_size = soft_local_max_size

        self.pad = self.soft_local_max_size // 2

    def forward(self, batch):
        b = batch.size(0)

        batch = F.relu(batch)  # [2,512,32,32]

        max_per_sample = torch.max(batch.view(b, -1), dim=1)[0]  # 每幅图特征的最大值[2,]

        exp = torch.exp(batch / max_per_sample.view(b, 1, 1, 1))  # [2,512,32,32]

        sum_exp = self.soft_local_max_size**2 * F.avg_pool2d(
            F.pad(exp, [self.pad] * 4, mode="constant", value=1.0),
            self.soft_local_max_size,  # (3,3)
            stride=1,
        )

        local_max_score = exp / sum_exp

        # 计算每个像素点在通道维度上的最大值 depth_wise_max，然后计算每个像素点相对于最大值的得分 depth_wise_max_score。
        depth_wise_max = torch.max(batch, dim=1)[0]  # 2,32,32
        depth_wise_max_score = batch / depth_wise_max.unsqueeze(1)

        all_scores = local_max_score * depth_wise_max_score

        score = torch.max(all_scores, dim=1)[0]

        score = score / torch.sum(score.view(b, -1), dim=1).view(b, 1, 1)
        # [2,32,32]

        return score


class D2Net(nn.Module):
    def __init__(self, model_file=None, use_cuda=True):
        super(D2Net, self).__init__()

        self.dense_feature_extraction = DenseFeatureExtractionModule(
            finetune_feature_extraction=True, use_cuda=use_cuda
        )

        self.detection = SoftDetectionModule()

        if model_file is not None:
            if use_cuda:
                self.load_state_dict(torch.load(model_file)["model"])
            else:
                self.load_state_dict(
                    torch.load(model_file, map_location="cpu")["model"]
                )

    def forward(self, batch):
        # batch size
        b = batch["image1"].size(0)

        # The output of torch.cat() is a tensor with dimension (2*batch_size, 3, 224, 224)
        dense_features = self.dense_feature_extraction(
            torch.cat([batch["image1"], batch["image2"]], dim=0)
        )

        scores = self.detection(dense_features)

        dense_features1 = dense_features[:b, :, :, :]
        dense_features2 = dense_features[b:, :, :, :]

        scores1 = scores[:b, :, :]
        scores2 = scores[b:, :, :]

        return {
            "dense_features1": dense_features1,
            "scores1": scores1,
            "dense_features2": dense_features2,
            "scores2": scores2,
        }
