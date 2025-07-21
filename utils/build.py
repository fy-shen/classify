import os
import warnings
from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets

from archs import CUSTOM_SET, get_torch_obj


class Builder:
    def __init__(self, cfg, gpu_id, logger):
        self.cfg = cfg
        self.gpu_id = gpu_id
        self.logger = logger
        self.weights = None

    def build_model(self, is_train=True):
        name = self.cfg.model
        pretrained = self.cfg.train.get("pretrained", False)
        obj = get_torch_obj(name, [models])
        if obj:
            if isinstance(pretrained, bool):
                weights = "DEFAULT" if pretrained else None
            elif isinstance(pretrained, str):
                weights = None if Path(pretrained).is_file() else pretrained.upper()
            else:
                warnings.warn(f"pretrained={pretrained} is not supported, use weights='DEFAULT' instead")
                weights = "DEFAULT"

            if weights is not None:
                try:
                    weight_enum = models.get_model_weights(obj)
                    self.weights = getattr(weight_enum, weights)
                    self.logger.log(f"Load weights for {name}: {self.weights}")
                except Exception as e:
                    warnings.warn(f"Failed to load weights enum for {name}: {e}")
                    weights = None
            model = obj(weights=weights)

            if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, self.cfg.num_classes)

            if isinstance(pretrained, str) and Path(pretrained).is_file():
                ckpt = torch.load(pretrained, map_location="cpu")
                model.load_state_dict(ckpt.get("state_dict", ckpt))

        # custom model
        elif name.lower() in CUSTOM_SET['model']:
            model_cfg = self.cfg.get("model_cfg", None)
            if model_cfg is None:
                raise ValueError(f"Custom model '{name}' requires `model_cfg` to be specified in config.")
            model = CUSTOM_SET['model'][name.lower()](OmegaConf.load(model_cfg))
            model_state = model.state_dict()
            # TODO: pretrain
            if is_train:
                if self.cfg.train.pretrained and Path(self.cfg.train.pretrained).is_file():
                    ckpt = torch.load(self.cfg.train.pretrained, map_location="cpu")
                    ckpt = ckpt.get("state_dict", ckpt)
                    sd = {}
                    for k, v in ckpt.items():
                        k = k[len('module.'):] if k.startswith('module.') else k
                        if 'fc' in k and k in model_state and v.shape != model_state[k].shape:
                            continue
                        sd[k] = v
                    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
                    self.logger.log_pretrain_msg(missing_keys, unexpected_keys)
            else:
                ckpt = torch.load(self.cfg.val.weight, map_location="cpu")
                ckpt = ckpt.get("model", ckpt)
                missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
                self.logger.log_pretrain_msg(missing_keys, unexpected_keys)
            return model
        else:
            raise ValueError(f"Model {name} is not supported.")

        return model

    def build_criterion(self):
        name = self.cfg.train.loss
        obj = get_torch_obj(name, [nn])
        if obj:
            return obj()

        # TODO: custom loss
        raise ValueError(f"Loss function {name} is not supported.")

    def build_optimizer(self, model):
        name = self.cfg.train.optimizer
        obj = get_torch_obj(name, [optim])
        if obj:
            args = OmegaConf.to_container(self.cfg.train.get("optim_params", {}), resolve=True)
            return obj(model.parameters(), **args)
        else:
            raise ValueError(f"Optimizer {name} is not supported.")

    def build_scheduler(self, optimizer):
        name = self.cfg.train.get("scheduler", None)
        if name is None:
            return None
        obj = get_torch_obj(name, [optim.lr_scheduler])
        if obj:
            args = OmegaConf.to_container(self.cfg.train.get("scheduler_params", {}), resolve=True)
            scheduler = obj(optimizer, **args)
            return scheduler
        else:
            raise ValueError(f"Scheduler {name} is not supported.")

    def build_dataset(self, split):
        is_train = split == "train"
        name = self.cfg.dataset
        trans = self.build_transform(is_train)
        obj = get_torch_obj(name, [datasets])
        if obj:
            data_root = self.cfg.get("data_root", None) or f'./datasets/{name}/'
            os.makedirs(data_root, exist_ok=True)
            try:
                return obj(root=data_root, split=split, transform=trans, download=True)
            except TypeError:
                pass
            try:
                return obj(root=data_root, train=is_train, transform=trans, download=True)
            except TypeError:
                pass
            raise ValueError(
                f"Dataset '{name}' does not support 'split' or 'train' keyword. "
            )
        # custom dataset
        elif name.lower() in CUSTOM_SET['dataset']:
            return CUSTOM_SET['dataset'][name.lower()](self.cfg, is_train, transform=trans)
        else:
            raise ValueError(f"Unknown Dataset {name}.")

    def build_transform(self, is_train):
        # 尝试加载指定 torchvision transform
        try:
            weight = eval(f"models.{self.cfg.data_trans}")
            return weight.transforms()
        except Exception:
            pass

        # custom transform
        if self.cfg.data_trans in CUSTOM_SET['transform']:
            return CUSTOM_SET['transform'][self.cfg.data_trans](self.cfg, is_train=is_train)

        # 使用 torchvision model 对应的 transform
        if self.weights is not None:
            return self.weights.transforms()
        else:
            raise ValueError(f"Dataset {self.cfg.dataset} has no matching transform.")


