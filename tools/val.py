import torch

from utils.distributed import set_env, setup_ddp, cleanup_ddp, rank_zero, reduce_tensor


def val_epoch(model, loader, criterion, gpu_id):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(gpu_id, non_blocking=True)
            targets = targets.to(gpu_id, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            total_correct += preds.eq(targets).sum().item()
            total_samples += inputs.size(0)

    # 多卡同步
    total_loss_tensor = reduce_tensor(torch.tensor(total_loss, device=gpu_id))
    total_correct_tensor = reduce_tensor(torch.tensor(total_correct, device=gpu_id))
    total_samples_tensor = reduce_tensor(torch.tensor(total_samples, device=gpu_id))

    if rank_zero():
        avg_loss = total_loss_tensor.item() / (total_samples_tensor.item())
        avg_acc = total_correct_tensor.item() / (total_samples_tensor.item())
        return avg_acc, avg_loss

    return None, None
