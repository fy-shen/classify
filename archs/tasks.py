import ast
from copy import deepcopy

import torch
import torch.nn as nn

from . import CUSTOM_MODULES


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model, self.save = parse_model(deepcopy(cfg))


def parse_model(model_dict):
    layers, save = [], []
    # from, number repeats, module, args
    for i, (f, n, m, args) in enumerate(model_dict["backbone"] + model_dict["head"]):
        m = getattr(torch.nn, m[3:]) if "nn." in m else CUSTOM_MODULES[m.lower()]
        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                except ValueError:
                    pass

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        layers.append(m_)
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist

    return nn.Sequential(*layers), sorted(save)
