import os
import numpy as np
import torch

from archs import register
from archs.evaluator import BaseEvaluator
from utils.distributed import reduce_tensor, gather_tensor


@register('evaluator')
class RegressPoint(BaseEvaluator):
    def __init__(self, gpu_id):
        super().__init__(gpu_id)
        self.W = 3840
        self.H = 1080
        self.thresholds = [5, 15, 30, 60, 120]
        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []
        self.loss = torch.tensor(0.0, device=self.gpu_id)
        self.samples = torch.tensor(0, device=self.gpu_id)

    def update(self, outputs, targets, loss, is_train):
        pred_coords, _ = outputs
        if not is_train:
            self.preds.append(pred_coords.detach())
            self.targets.append(targets.detach())

        self.loss += loss.detach() * pred_coords.size(0)
        self.samples += pred_coords.size(0)

    def get_current_metrics(self):
        avg_loss = self.loss / (self.samples + 1e-8)
        return {"loss": f"{avg_loss:.4f}"}

    def synchronize(self, is_train):
        reduced_loss = reduce_tensor(self.loss)
        reduced_count = reduce_tensor(self.samples)

        avg_loss = reduced_loss.item() / (reduced_count.item() + 1e-8)
        avg_acc = 0.0
        if not is_train:
            preds = torch.cat(self.preds)
            targets = torch.cat(self.targets)
            self.preds_tensor = gather_tensor(preds)
            self.targets_tensor = gather_tensor(targets)

            preds_px = self.preds_tensor.cpu().numpy() * [self.W, self.H]
            targets_px = self.targets_tensor.cpu().numpy() * [self.W, self.H]
            dists = np.linalg.norm(preds_px - targets_px, axis=1)

            accs = [(dists <= th).mean() for th in self.thresholds]
            avg_acc = np.mean(accs)

        return avg_loss, avg_acc

    def log_metrics(self, logger, cfg):
        preds = self.preds_tensor.cpu().numpy()
        targets = self.targets_tensor.cpu().numpy()

        preds_px = preds * [self.W, self.H]
        targets_px = targets * [self.W, self.H]
        dists = np.linalg.norm(preds_px - targets_px, axis=1)
        logger.log(f"{'MeanDist':<10}{dists.mean():.3f}")

        logger.log(f"{'Threshold':<12}{'Accuracy':<12}{'Num'}")
        for th in self.thresholds:
            acc = (dists <= th).mean()
            correct = (dists <= th).sum()
            logger.log(f"<= {th:<4}px   {acc:<12.3%}{correct}")
