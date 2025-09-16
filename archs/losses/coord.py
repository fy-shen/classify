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
    def __init__(self, lambda_coord, lambda_heatmap, sigma):
        super().__init__()

        self.lambda_coord = lambda_coord
        self.lambda_heatmap = lambda_heatmap
        self.sigma = sigma

    def forward(self, outputs, targets):
        pred_coords, pred_heatmaps = outputs
        _, h, w = pred_heatmaps.shape
        target_coords = targets
        target_heatmaps = gen_gaussian_heatmaps(target_coords, h, w, self.sigma)

        loss_coord = F.l1_loss(pred_coords, target_coords)
        loss_heatmap = F.mse_loss(pred_heatmaps, target_heatmaps)
        loss = self.lambda_coord * loss_coord + self.lambda_heatmap * loss_heatmap
        return loss
