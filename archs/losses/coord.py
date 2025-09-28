import torch
import torch.nn as nn
import torch.nn.functional as F

from archs import register


def gen_gaussian_heatmaps(coords, h, w, sigma=1.5, device=None):
    """
    批量生成高斯GT热力图
    """
    b = coords.size(0)
    device = device if device is not None else coords.device

    # 构建网格 (H, W)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )
    grid_x = grid_x.float() / (w - 1)  # 归一化到 [0,1]
    grid_y = grid_y.float() / (h - 1)

    # 扩展到 batch 维度 (b, H, W)
    grid_x = grid_x.unsqueeze(0).expand(b, -1, -1)  # (b,H,W)
    grid_y = grid_y.unsqueeze(0).expand(b, -1, -1)

    # coords: (b,2) → (b,1,1)
    x = coords[:, 0].view(b, 1, 1)
    y = coords[:, 1].view(b, 1, 1)

    # 高斯分布
    heatmaps = torch.exp(-((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * sigma ** 2))

    # 归一化 (每个样本单独归一化)
    heatmaps = heatmaps / (heatmaps.sum(dim=(1, 2), keepdim=True) + 1e-8)

    return heatmaps


@register('loss')
class CoordHeatmapLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.lambda_coord = cfg.train.loss_params.lambda_coord
        self.lambda_heatmap = cfg.train.loss_params.lambda_heatmap
        self.sigma = cfg.train.loss_params.sigma

    def forward(self, outputs, targets):
        pred_coords, pred_heatmaps = outputs
        _, h, w = pred_heatmaps.shape
        target_coords = targets
        target_heatmaps = gen_gaussian_heatmaps(target_coords, h, w, self.sigma)

        # 坐标回归损失
        # loss_coord = F.l1_loss(pred_coords, target_coords)
        loss_coord = F.smooth_l1_loss(pred_coords, target_coords, beta=0.1)
        # heatmap损失
        # loss_heatmap = F.mse_loss(pred_heatmaps, target_heatmaps)
        loss_heatmap = bec_loss(pred_heatmaps, target_heatmaps)
        # loss_heatmap = focal_loss(pred_heatmaps, target_heatmaps, alpha=0.25, gamma=2.0, reduction='sum')

        lambda_coord = 1.0 / (loss_coord.detach() + 1e-8)
        lambda_heatmap = 1.0 / (loss_heatmap.detach() + 1e-8)
        lambda_coord = lambda_coord / (lambda_coord + lambda_heatmap)
        lambda_heatmap = lambda_heatmap / (lambda_coord + lambda_heatmap)
        loss = lambda_coord * loss_coord + lambda_heatmap * loss_heatmap
        # loss = self.lambda_coord * loss_coord + self.lambda_heatmap * loss_heatmap
        return loss


def bec_loss(pred, target):
    pos_mask = target > 0.01  # 目标区域
    neg_mask = ~pos_mask

    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

    loss_pos = (bce * pos_mask).sum() / (pos_mask.sum() + 1e-6)
    loss_neg = (bce * neg_mask).sum() / (neg_mask.sum() + 1e-6)

    loss = 0.6 * loss_pos + 0.4 * loss_neg
    return loss


def focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    pred: (B, H, W) 或 (B, 1, H, W)，未经过 sigmoid
    target: (B, H, W)，取值范围 [0,1]，通常是高斯分布
    """
    # sigmoid
    pred_sigmoid = torch.sigmoid(pred)

    # 交叉熵
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

    # p_t = p  if y=1  else (1-p)
    p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)

    # focal weight
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * ((1 - p_t) ** gamma)

    # loss
    loss = focal_weight * ce_loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
