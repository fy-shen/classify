import os
import random
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
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
    os.environ["PYTHONHASHSEED"] = str(seed)
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


class Logger:
    def __init__(self, cfg):
        self.log_path = os.path.join(cfg.save_dir, 'log.txt')
        if cfg.train.get("resume", False):
            with open(self.log_path, "a") as f:
                f.write("\n\nResuming from checkpoint...\n")
        else:
            with open(self.log_path, "w") as f:
                f.write(f"Training {cfg.model} on dataset {cfg.dataset}\n")

    @staticmethod
    def make_separator(title, width=60, fill='='):
        if title:
            title = f" {title} "
        return f"{title:{fill}^{width}}"

    def log(self, msg, print_out=True):
        if print_out:
            # tqdm.write(msg)
            print(msg)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    def log_train_tools(self, model, optimizer, scheduler, cfg):
        msg = '\n' + self.make_separator("Model") + '\n'
        msg += str(model)
        msg += '\n' + self.make_separator("Optimizer") + '\n'
        msg += f"{type(optimizer).__name__} | Params: {sum(p.numel() for p in model.parameters())}"
        if scheduler:
            msg += '\n' + self.make_separator("Scheduler") + '\n'
            msg += f"{type(scheduler).__name__} | Params: {cfg.train.get('scheduler_params', {})}"
        msg += '\n' + self.make_separator("")
        self.log(msg)

    def log_cfg(self, cfg):
        msg = '\n' + self.make_separator("Config") + '\n'
        msg += OmegaConf.to_yaml(cfg, resolve=True)
        msg += '\n' + self.make_separator("")
        self.log(msg)
