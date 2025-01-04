# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()

        # 加载预训练的VGG19模型
        vgg = models.vgg19(pretrained=True).features

        # 冻结所有VGG参数
        for param in vgg.parameters():
            param.requires_grad = False

        # 将VGG模块分割为多个块
        self.blocks = nn.ModuleList([
            vgg[:4],  # conv1_2
            vgg[4:9],  # conv2_2
            vgg[9:18],  # conv3_4
            vgg[18:27],  # conv4_4
            vgg[27:36]  # conv5_4
        ])

        # 是否需要调整输入图像大小
        self.resize = resize

        # 图像预处理
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 移动模型到GPU(如果可用)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def normalize(self, x):
        """归一化输入图像"""
        return (x - self.mean) / self.std

    def forward(self, x, y, weights=None):
        """
        计算感知损失
        Args:
            x (tensor): 生成的图像
            y (tensor): 目标图像
            weights (list): 每层特征的权重，默认为None表示所有层权重相等
        """
        if weights is None:
            weights = [1.0] * len(self.blocks)

        # 确保输入在[0,1]范围内
        if x.max() > 1 or x.min() < 0:
            x = torch.clamp(x, 0, 1)
        if y.max() > 1 or y.min() < 0:
            y = torch.clamp(y, 0, 1)

        # 调整图像大小(如果需要)
        if self.resize:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = nn.functional.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)

        # 归一化
        x = self.normalize(x)
        y = self.normalize(y)

        # 计算每个块的特征损失
        loss = 0
        x_feat = x
        y_feat = y

        for block, weight in zip(self.blocks, weights):
            x_feat = block(x_feat)
            y_feat = block(y_feat)

            # 使用L2损失计算特征图的差异
            loss += weight * nn.functional.mse_loss(x_feat, y_feat)

        return loss
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target))*2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1 - dice
        return loss


class BinaryFocalLoss(nn.Module):
    """
    参考 https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=2.0, alpha=0.25, epsilon=1.e-9):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        multi_hot_key = target
        logits = input
        # 如果模型没有做sigmoid的话，这里需要加上
        # logits = torch.sigmoid(logits)
        zero_hot_key = 1 - multi_hot_key
        loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()
