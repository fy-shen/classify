import os
import random
import cv2
import logging
import numpy as np
from omegaconf import OmegaConf
from prettytable import PrettyTable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

from utils.distributed import rank_zero


class Video:
    def __init__(self, video_path):
        self.video_path = video_path

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            logging.warning(f'Open Video {video_path} failed')

        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))


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
        self.save_dir = cfg.save_dir
        self.log_path = os.path.join(cfg.save_dir, 'log.txt')
        self.history = {}
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
        if not rank_zero():
            return
        if print_out:
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

    def log_pretrain_msg(self, missing_keys, unexpected_keys):
        msg = '\n' + self.make_separator("Pretrain") + '\n'

        t1 = PrettyTable(["Missing Keys"])
        t1.align["Missing Keys"] = "l"
        for k in missing_keys:
            t1.add_row([k])

        t2 = PrettyTable(["Unexpected Keys"])
        t2.align["Unexpected Keys"] = "l"
        for k in unexpected_keys:
            t2.add_row([k])
        msg += t1.get_string()
        msg += '\n' + t2.get_string()
        msg += '\n' + self.make_separator("")
        self.log(msg)

    def update_history(self, phase, metrics):
        for key, value in metrics.items():
            name = f"{phase}_{key}"
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)

    def plot_history(self):
        self.plot_pic('loss', 'Loss', 'Train and Val Loss', 'loss.png')
        self.plot_pic('acc', 'Acc', 'Train and Val Acc', 'acc.png')

    def plot_pic(self, key, ylabel, title, fn):
        plt.figure(figsize=(10, 6))
        for phase in ['train', 'val']:
            y_key = f"{phase}_{key}"
            x_key = f"{phase}_epoch"
            if y_key in self.history:
                y = self.history[y_key]
                x = self.history[x_key]
                plt.plot(x, y, label=y_key)

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        loss_path = os.path.join(self.save_dir, fn)
        plt.savefig(loss_path)
        plt.close()
