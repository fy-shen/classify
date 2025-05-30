import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from utils import set_random_seed
from utils.distributed import set_env, setup_ddp, cleanup_ddp


def train_worker(rank, cfg):
    set_random_seed(cfg.SEED, cfg.DETERMINISTIC)
    gpu_id = cfg.GPU_IDS[rank]
    model = build_model(cfg).to(gpu_id)
    if cfg.GPU_NUM > 1:
        setup_ddp(rank, cfg.GPU_NUM, cfg.GPU_IDS)
        model = DDP(model, device_ids=[gpu_id])

    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    dataset_train = build_dataset(cfg)
    if cfg.GPU_NUM > 1:
        sampler_train = DistributedSampler(dataset_train, num_replicas=cfg.GPU_NUM, rank=rank, shuffle=True, drop_last=True)
    else:
        sampler_train = None
    loader_train = DataLoader(
        dataset_train,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=sampler_train is None,
    )

    for epoch in range(cfg.epochs):
        if sampler_train is not None:
            sampler_train.set_epoch(epoch)
        train_epoch(model, loader_train, criterion, optimizer, scheduler, epoch, cfg)

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
