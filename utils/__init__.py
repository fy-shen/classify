import os
import random
import numpy as np
import torch


def set_random_seed(seed: int, deterministic: bool = False):
    """
    设置随机种子以确保结果可复现。

    Args:
        seed (int): 随机种子值。
        deterministic (bool): 是否设置 PyTorch 为确定性模式（可能影响性能）。
    """
    # 出于安全考虑（哈希冲突攻击），Python 哈希操作的种子是随机生成的，例如遍历字典或集合时的顺序不同
    # PYTHONHASHSEED 只对 当前 Python 进程中创建的哈希函数有效，若使用 multiprocessing 子进程中也要显式设置
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确定性模式（Deterministic Mode）在相同的输入、代码和运行环境下，每次运行都能产生完全一致的输出结果
    # PyTorch 中很多操作，例如卷积、池化、dropout等默认使用 非确定性算法，计算速度更快但会引入微小的不确定性
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
