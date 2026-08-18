import os
import socket
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


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
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(gpu_ids[rank])


def cleanup_ddp():
    # 销毁分布式环境
    dist.destroy_process_group()


def rank_zero():
    # 多卡时，判断是否为 rank 0
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_tensor(tensor, op=dist.ReduceOp.SUM):
    # 聚合多卡结果
    if dist.is_initialized():
        rt = tensor.clone()
        dist.all_reduce(rt, op=op)
        return rt
    else:
        return tensor


def sync_module_buffers(module, src=0):
    # DDP eval 绕过 wrapper 时，先同步 BN 等 buffer，保证各 rank 推理状态一致。
    if dist.is_initialized():
        for buffer in module.buffers():
            dist.broadcast(buffer, src=src)


def gather_tensor(tensor, dst=0):
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # 先把每个 rank 的长度广播出来
        local_size = torch.tensor([tensor.size(0)], device=tensor.device)
        sizes = [torch.zeros(1, device=tensor.device, dtype=torch.long) for _ in range(world_size)]
        dist.all_gather(sizes, local_size)
        sizes = [int(s.item()) for s in sizes]

        # pad 到最大长度
        max_size = max(sizes)
        padded = torch.zeros(max_size, dtype=tensor.dtype, device=tensor.device)
        padded[:tensor.size(0)] = tensor

        gather_list = None
        if rank == dst:
            gather_list = [torch.zeros_like(padded) for _ in range(world_size)]
        dist.gather(padded, gather_list=gather_list, dst=dst)

        # 去掉 pad
        if rank == dst:
            result = []
            for g, sz in zip(gather_list, sizes):
                result.append(g[:sz])
            return torch.cat(result, dim=0)
        else:
            return None
    else:
        return tensor


class DistributedEvalSampler(Sampler):
    """
    验证/测试用分布式采样器：按 rank 切分索引，不补齐样本，避免重复样本污染指标。
    """
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.indices = list(range(len(dataset)))[rank::num_replicas]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
