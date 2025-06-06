import os
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from utils import set_random_seed
from utils.build import Builder
from utils.distributed import set_env, setup_ddp, cleanup_ddp, rank_zero, reduce_tensor
from tools.val import val_epoch


def train_epoch(model, loader, criterion, optimizer, scheduler, epoch, gpu_id, cfg):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for step, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(gpu_id, non_blocking=True)
        targets = targets.to(gpu_id, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        total_correct += preds.eq(targets).sum().item()
        total_samples += inputs.size(0)

    reduced_loss = reduce_tensor(torch.tensor(total_loss, device=gpu_id))
    reduced_acc = reduce_tensor(torch.tensor(total_correct, device=gpu_id))
    reduced_count = reduce_tensor(torch.tensor(total_samples, device=gpu_id))
    if rank_zero():
        avg_loss = reduced_loss.item() / reduced_count.item()
        avg_acc = reduced_acc.item() / reduced_count.item()
        print(f"[Epoch {epoch:>03d}] Loss: {avg_loss:.4f} | Acc: {avg_acc:.2%}")

    if scheduler is not None:
        scheduler.step()


def train_worker(rank, cfg):
    set_random_seed(cfg.SEED, cfg.DETERMINISTIC)
    gpu_id = cfg.GPU_IDS[rank]

    builder = Builder(cfg, gpu_id)
    model = builder.build_model().to(gpu_id)
    if cfg.GPU_NUM > 1:
        setup_ddp(rank, cfg.GPU_NUM, cfg.GPU_IDS)
        model = DDP(model, device_ids=[gpu_id])

    criterion = builder.build_criterion()
    optimizer = builder.build_optimizer(model)
    scheduler = builder.build_scheduler(optimizer)

    dataset_train = builder.build_dataset('train')
    dataset_val = builder.build_dataset('val')
    if cfg.GPU_NUM > 1:
        sampler_train = DistributedSampler(dataset_train, num_replicas=cfg.GPU_NUM, rank=rank, shuffle=True, drop_last=True)
        sampler_val = DistributedSampler(dataset_val, num_replicas=cfg.GPU_NUM, rank=rank, shuffle=False, drop_last=False)
    else:
        sampler_train = None
        sampler_val = None
    loader_train = DataLoader(
        dataset_train,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=sampler_train,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=sampler_val,
    )

    best_acc = 0.0
    start_epoch = 0
    if cfg.train.get("resume", False) and Path(cfg.train.resume_path).is_file():
        checkpoint = torch.load(cfg.train.resume_path, map_location=f"cuda:{gpu_id}")
        state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])

        best_acc = checkpoint.get('best_acc', 0.0)
        start_epoch = checkpoint.get('epoch', 0) + 1
        if rank_zero():
            lr = optimizer.param_groups[0]["lr"]
            print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2%}, lr={lr}")

    for epoch in range(start_epoch, cfg.epochs):
        if sampler_train is not None:
            sampler_train.set_epoch(epoch)
        train_epoch(model, loader_train, criterion, optimizer, scheduler, epoch, gpu_id, cfg)
        ckpt = {
            "epoch": epoch,
            "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "best_acc": best_acc,
        }
        if (epoch + 1) % cfg.EVAL_PERIOD == 0:
            acc = val_epoch(model, loader_val, criterion, gpu_id, epoch)
            if rank_zero() and acc is not None:
                if acc > best_acc:
                    best_acc = acc
                    ckpt["best_acc"] = best_acc
                    torch.save(ckpt, os.path.join(cfg.save_dir, "best.pth"))

        torch.save(ckpt, os.path.join(cfg.save_dir, "last.pth"))
    if cfg.GPU_NUM > 1:
        cleanup_ddp()


def train(cfg):
    if cfg.GPU_NUM > 1:
        set_env()
        mp.spawn(
            train_worker,
            args=(cfg, ),
            nprocs=cfg.GPU_NUM
        )
    else:
        train_worker(0, cfg)
