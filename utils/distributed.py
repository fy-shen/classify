import os
import socket
import torch
import torch.distributed as dist


def find_free_port():
    # 获取一个当前主机上的可用端口
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 0))
        return s.getsockname()[1]


def set_env():
    # master_port: 通信端口号（不同任务可修改避免冲突）
    master_port = str(find_free_port())
    os.environ['MASTER_PORT'] = master_port
    os.environ['MASTER_ADDR'] = '127.0.0.1'


def setup_ddp(rank, world_size, gpu_ids):
    """
    初始化分布式训练环境

    Args:
        rank: 当前进程编号（0 ~ world_size-1）
        world_size: 总进程数（总 GPU 数）
        gpu_ids: GPU 列表，如 [0, 1, 2, 3]
    """
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(gpu_ids[rank])


def cleanup_ddp():
    # 销毁分布式环境
    dist.destroy_process_group()


def rank_zero():
    # 多卡时，判断是否为 rank 0
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def reduce_tensor(tensor, op=torch.distributed.ReduceOp.SUM):
    # 聚合多卡结果
    if torch.distributed.is_initialized():
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=op)
        return rt
    else:
        return tensor
