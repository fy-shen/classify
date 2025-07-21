import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from tools.train import run_epoch
from utils import set_random_seed, Logger
from utils.build import Builder
from utils.distributed import set_env, setup_ddp, cleanup_ddp, rank_zero, reduce_tensor


def val_worker(rank, cfg):
    logger = None
    if rank_zero():
        logger = Logger(cfg)
        logger.log_cfg(cfg)

    set_random_seed(cfg.SEED, cfg.DETERMINISTIC)
    gpu_id = cfg.GPU_IDS[rank]

    builder = Builder(cfg, gpu_id, logger)
    model = builder.build_model(False).to(gpu_id)
    if cfg.GPU_NUM > 1:
        setup_ddp(rank, cfg.GPU_NUM, cfg.GPU_IDS)
        model = DDP(model, device_ids=[gpu_id])

    criterion = builder.build_criterion()
    dataset = builder.build_dataset('val')
    if cfg.GPU_NUM > 1:
        sampler = DistributedSampler(dataset, num_replicas=cfg.GPU_NUM, rank=rank, shuffle=False, drop_last=False)
    else:
        sampler = None
    loader_val = DataLoader(
        dataset,
        batch_size=cfg.val.batchsize,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    loss, acc = run_epoch(model, loader_val, criterion, gpu_id, is_train=False)
    if rank_zero():
        logger.log(f'Val: Loss={loss:.3f}, Acc={acc:.2%}')

    if cfg.GPU_NUM > 1:
        cleanup_ddp()


def val(cfg):
    if cfg.GPU_NUM > 1:
        set_env()
        mp.spawn(
            val_worker,
            args=(cfg, ),
            nprocs=cfg.GPU_NUM
        )
    else:
        val_worker(0, cfg)
