import os
import torch

from archs import register
from archs.evaluator import BaseEvaluator
from utils.file import load_label_map
from utils.distributed import reduce_tensor, gather_tensor


@register('evaluator')
class Classify(BaseEvaluator):
    def __init__(self, gpu_id):
        super().__init__(gpu_id)

        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []
        self.loss = torch.tensor(0.0, device=self.gpu_id)
        self.correct = torch.tensor(0, device=self.gpu_id)
        self.samples = torch.tensor(0, device=self.gpu_id)

    def update(self, outputs, targets, loss, is_train):
        preds = outputs.argmax(dim=1)
        if not is_train:
            self.preds.append(preds.detach())
            self.targets.append(targets.detach())

        self.loss += loss.detach() * outputs.size(0)
        self.correct += preds.eq(targets).sum()
        self.samples += outputs.size(0)

    def get_current_metrics(self):
        avg_loss = self.loss / (self.samples + 1e-8)
        avg_acc = self.correct / (self.samples + 1e-8)
        return {"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.3%}"}

    def synchronize(self, is_train):
        reduced_loss = reduce_tensor(self.loss)
        reduced_acc = reduce_tensor(self.correct)
        reduced_count = reduce_tensor(self.samples)

        sample_count = reduced_count.item()
        avg_loss = reduced_loss.item() / sample_count if sample_count else 0.0
        avg_acc = reduced_acc.item() / sample_count if sample_count else 0.0

        if not is_train:
            if self.preds:
                preds = torch.cat(self.preds)
                targets = torch.cat(self.targets)
            else:
                preds = torch.empty(0, dtype=torch.long, device=self.gpu_id)
                targets = torch.empty(0, dtype=torch.long, device=self.gpu_id)
            # 所有 rank 都需进入 gather，rank0 汇总完整且不重复的验证结果。
            self.preds_tensor = gather_tensor(preds)
            self.targets_tensor = gather_tensor(targets)

        return avg_loss, avg_acc

    def log_metrics(self, logger, cfg):
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
        class_map = cfg.data_params.get('class_map', None)
        if class_map:
            label_map = load_label_map(os.path.join(cfg.data_params.root_path, class_map))
        else:
            label_map = {i: name for i, name in enumerate(cfg.get('class_names', []))}
            if not label_map:
                label_map = {i: str(i) for i in range(cfg.num_classes)}
        cls_names = [label_map[i] for i in range(len(label_map))]
        cls_num = len(cls_names)
        preds, targets = self.preds_tensor.cpu().numpy(), self.targets_tensor.cpu().numpy()

        logger.log(f"{'Class':<15}{'TP':<8}{'FP':<8}{'P':<8}{'R':<8}{'F1':<8}{'Support'}")
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, labels=list(range(cls_num))
        )
        cm = confusion_matrix(targets, preds, labels=list(range(cls_num)))
        for i, name in enumerate(cls_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            logger.log(f"{name:<15}{tp:<8}{fp:<8}{precision[i]:<8.3f}{recall[i]:<8.3f}{f1[i]:<8.3f}{support[i]}")
        return cm
