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
from utils.distributed import rank_zero


class Builder:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.weights = None

    def load_weights(self, model, weight_path, is_train=False):
        state_dict = model.get_state_dict(weight_path, is_train) if hasattr(model, 'get_state_dict') else \
            torch.load(weight_path, weights_only=False)
        state_dict = state_dict.get('model', state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if rank_zero():
            self.logger.log_pretrain_msg(missing_keys, unexpected_keys)

    def build_model(self, mode):
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

            if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear) and self.cfg.get('num_classes'):
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

            # TODO: pretrain
            if mode == 'train':
                if self.cfg.train.pretrained and Path(self.cfg.train.pretrained).is_file():
                    weight_path = self.cfg.train.pretrained
                    self.load_weights(model, weight_path, True)
            elif mode == 'val':
                weight_path = self.cfg.val.weight
                self.load_weights(model, weight_path)
            elif mode == 'test':
                weight_path = self.cfg.test.weight
                self.load_weights(model, weight_path)
            else:
                raise ValueError(f"Build Model Mode {name} is not supported.")
            return model
        else:
            raise ValueError(f"Model {name} is not supported.")

        return model

    def build_criterion(self):
        name = self.cfg.train.loss
        obj = get_torch_obj(name, [nn])
        if obj:
            args = OmegaConf.to_container(self.cfg.train.get("loss_params") or OmegaConf.create({}), resolve=True)
            if args.get("weight", None) is not None:
                args["weight"] = torch.FloatTensor(args["weight"])
            return obj(**args)

        # TODO: custom loss
        elif name.lower() in CUSTOM_SET['loss']:
            return CUSTOM_SET['loss'][name.lower()](self.cfg)
        raise ValueError(f"Loss function {name} is not supported.")

    def build_optimizer(self, model):
        name = self.cfg.train.optimizer
        obj = get_torch_obj(name, [optim])
        if obj:
            args = OmegaConf.to_container(self.cfg.train.get("optim_params") or OmegaConf.create({}), resolve=True)
            policies = model.get_optim_policies() if hasattr(model, 'get_optim_policies') else model.parameters()
            return obj(policies, **args)
        else:
            raise ValueError(f"Optimizer {name} is not supported.")

    def build_scheduler(self, optimizer):
        name = self.cfg.train.get("scheduler", None)
        if name is None:
            return None
        obj = get_torch_obj(name, [optim.lr_scheduler])
        if obj:
            args = OmegaConf.to_container(self.cfg.train.get("scheduler_params") or OmegaConf.create({}), resolve=True)
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

    def build_evaluator(self, gpu_id):
        name = self.cfg.evaluator
        if name.lower() in CUSTOM_SET['evaluator']:
            return CUSTOM_SET['evaluator'][name.lower()](gpu_id)
        else:
            raise ValueError(f"Evaluator {name} is not supported.")


