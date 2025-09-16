import os
import time
from pathlib import Path
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from utils import set_random_seed, Logger
from utils.build import Builder
from utils.distributed import set_env, setup_ddp, cleanup_ddp, rank_zero


def run_epoch(
        model,
        loader,
        criterion,
        evaluator,
        gpu_id,
        optimizer=None,
        scheduler=None,
        epoch=None,
        is_train=True,
):
    model.train() if is_train else model.eval()

    # total_loss = torch.tensor(0.0, device=gpu_id)
    # total_correct = torch.tensor(0, device=gpu_id)
    # total_samples = torch.tensor(0, device=gpu_id)
    #
    # all_preds, all_targets, preds_tensor, targets_tensor = [], [], None, None
    if evaluator is not None:
        evaluator.reset()

    desc = f"[Train Epoch {epoch:>3d}]" if is_train else f"[Val]"
    pbar = tqdm(loader, desc=desc, ncols=100) if rank_zero() else loader
    with torch.set_grad_enabled(is_train):
        for step, (inputs, targets) in enumerate(pbar):
            # print(inputs.shape)
            inputs = inputs.to(gpu_id, non_blocking=True)
            targets = targets.to(gpu_id, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if is_train:
                loss.backward()
                optimizer.step()

            evaluator.update(outputs.detach(), targets.detach(), loss, is_train)
            if rank_zero():
                pbar.set_postfix(evaluator.get_current_metrics())

        if is_train and scheduler is not None:
            scheduler.step()

        avg_loss, avg_metric = evaluator.synchronize(is_train)
        return avg_loss, avg_metric


def train_worker(rank, cfg):
    logger = Logger(cfg) if rank_zero() else None
    logger.log_cfg(cfg) if rank_zero() else None
    set_random_seed(cfg.SEED, cfg.DETERMINISTIC)
    gpu_id = cfg.GPU_IDS[rank]

    start_time = time.time()

    builder = Builder(cfg, logger)
    model = builder.build_model('train').to(gpu_id)
    if cfg.GPU_NUM > 1:
        setup_ddp(rank, cfg.GPU_NUM, cfg.GPU_IDS)
        model = DDP(model, device_ids=[gpu_id])

    criterion = builder.build_criterion().to(gpu_id)
    optimizer = builder.build_optimizer(model)
    scheduler = builder.build_scheduler(optimizer)
    evaluator = builder.build_evaluator(gpu_id)
    if rank_zero():
        logger.log_train_tools(model, optimizer, scheduler, cfg)

    dataset_train = builder.build_dataset('train')
    dataset_val = builder.build_dataset('val')
    if cfg.GPU_NUM > 1:
        sampler_train = DistributedSampler(dataset_train, num_replicas=cfg.GPU_NUM, rank=rank, shuffle=True, drop_last=True)
        sampler_val = DistributedSampler(dataset_val, num_replicas=cfg.GPU_NUM, rank=rank, shuffle=False, drop_last=False)
        shuffle = False
    else:
        sampler_train = None
        sampler_val = None
        shuffle = True
    loader_train = DataLoader(
        dataset_train,
        batch_size=cfg.train.batchsize,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=sampler_train,
        shuffle=shuffle
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=cfg.train.batchsize,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=sampler_val,
    )

    best_metric = 0.0
    start_epoch = 0
    best_ckpt = {
        "epoch": -1,
        "train_loss": None,
        "train_metric": None,
        "val_loss": None,
        "val_metric": None
    }
    if cfg.train.get("resume", False):
        if Path(cfg.train.resume_path).is_file():
            checkpoint = torch.load(cfg.train.resume_path, map_location=f"cuda:{gpu_id}")
            state_dict = checkpoint.get('model', checkpoint)
            model.load_state_dict(state_dict)
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint and scheduler:
                scheduler.load_state_dict(checkpoint['scheduler'])

            best_metric = checkpoint.get('best_metric', 0.0)
            start_epoch = checkpoint.get('epoch', 0) + 1
            if rank_zero():
                lr = optimizer.param_groups[0]["lr"]
                logger.log(f"Resumed from epoch {start_epoch}, best_metric={best_metric:.2%}, lr={lr}")
                if 'history' in checkpoint:
                    logger.history = checkpoint['history']
        else:
            raise FileNotFoundError(f"No such pretrain file: {cfg.train.resume_path}")

    if rank_zero():
        logger.log("\nStart training...")
    for epoch in range(start_epoch, cfg.train.epochs):
        if sampler_train is not None:
            sampler_train.set_epoch(epoch)
        train_loss, train_metric = run_epoch(
            model, loader_train, criterion, evaluator, gpu_id, optimizer, scheduler, epoch, is_train=True
        )
        if rank_zero():
            lr = optimizer.param_groups[0]["lr"]
            logger.log(f"Epoch {epoch:>3d} | Train: Loss={train_loss:.3f} | Metric={train_metric:.2%} | LR={lr:.4g}", False)
        logger.update_history('train', {'epoch': epoch, 'loss': train_loss, 'metric': train_metric})
        ckpt = {
            "epoch": epoch,
            "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "best_metric": best_metric,
        }
        # evaluate every EVAL_PERIOD epochs
        if (epoch + 1) % cfg.EVAL_PERIOD == 0:
            val_loss, val_metric = run_epoch(
                model, loader_val, criterion, evaluator, gpu_id, is_train=False
            )
            logger.update_history('val', {'epoch': epoch, 'loss': val_loss, 'metric': val_metric})
            if rank_zero() and val_metric is not None:
                ckpt["history"] = logger.history
                logger.log(f"          | Val  : Loss={val_loss:.3f} | Metric={val_metric:.2%}", False)
                if val_metric >= best_metric:
                    best_metric = val_metric
                    ckpt["best_metric"] = best_metric
                    torch.save(ckpt, os.path.join(cfg.save_dir, "best.pth"))
                    best_ckpt.update({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_metric": train_metric,
                        "val_loss": val_loss,
                        "val_metric": val_metric
                    })

        torch.save(ckpt, os.path.join(cfg.save_dir, "last.pth"))

    if rank_zero():
        end_time = time.time()
        duration = end_time - start_time
        h = int(duration // 3600)
        m = int((duration % 3600) // 60)
        s = int(duration % 60)
        logger.log(f"\nTraining finished in {h:02d}:{m:02d}:{s:02d}")

        if best_ckpt["epoch"] >= 0:
            logger.log(f"Best Model Info:")
            logger.log(f"  Epoch        : {best_ckpt['epoch']}")
            logger.log(f"  Train Loss   : {best_ckpt['train_loss']:.4f}")
            logger.log(f"  Train Metric : {best_ckpt['train_metric']:.2%}")
            logger.log(f"  Val Loss     : {best_ckpt['val_loss']:.4f}")
            logger.log(f"  Val Metric   : {best_ckpt['val_metric']:.2%}")

        logger.plot_history()
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
