import os
import warnings
from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets

from archs import ensure_registered, get_torch_obj
from utils.distributed import rank_zero


class Builder:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.weights = None

    @staticmethod
    def _get_custom(kind, name):
        return ensure_registered(kind, name)

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
                    if self.logger is not None and rank_zero():
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
        elif self._get_custom('model', name):
            model_cfg = self.cfg.get("model_cfg", None)
            if model_cfg is None:
                raise ValueError(f"Custom model '{name}' requires `model_cfg` to be specified in config.")
            model = self._get_custom('model', name)(OmegaConf.load(model_cfg))

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
        elif self._get_custom('loss', name):
            return self._get_custom('loss', name)(self.cfg)
        raise ValueError(f"Loss function {name} is not supported.")

    def build_optimizer(self, model):
        name = self.cfg.train.optimizer
        obj = get_torch_obj(name, [optim])
        if obj:
            args = OmegaConf.to_container(self.cfg.train.get("optim_params") or OmegaConf.create({}), resolve=True)
            if hasattr(model, 'get_optim_policies'):
                policies = model.get_optim_policies()
            elif hasattr(model, 'module') and hasattr(model.module, 'get_optim_policies'):
                policies = model.module.get_optim_policies()
            else:
                policies = model.parameters()
            policies = self._apply_optim_policy_multipliers(policies, args)
            return obj(policies, **args)
        else:
            raise ValueError(f"Optimizer {name} is not supported.")

    @staticmethod
    def _apply_optim_policy_multipliers(policies, optim_args):
        if not isinstance(policies, list) or not all(isinstance(p, dict) for p in policies):
            return policies

        base_lr = optim_args.get("lr")
        base_weight_decay = optim_args.get("weight_decay")
        param_groups = []
        for group in policies:
            group = group.copy()
            lr_mult = group.pop("lr_mult", None)
            decay_mult = group.pop("decay_mult", None)
            if lr_mult is not None and base_lr is not None:
                group["lr"] = base_lr * lr_mult
            if decay_mult is not None and base_weight_decay is not None:
                group["weight_decay"] = base_weight_decay * decay_mult
            param_groups.append(group)
        return param_groups

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
        elif self._get_custom('dataset', name):
            return self._get_custom('dataset', name)(self.cfg, is_train, transform=trans)
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
        trans = self._get_custom('transform', self.cfg.data_trans)
        if trans:
            return trans(self.cfg, is_train=is_train)

        # 使用 torchvision model 对应的 transform
        if self.weights is not None:
            return self.weights.transforms()
        else:
            raise ValueError(f"Dataset {self.cfg.dataset} has no matching transform.")

    def build_evaluator(self, gpu_id):
        name = self.cfg.evaluator
        evaluator = self._get_custom('evaluator', name)
        if evaluator:
            return evaluator(gpu_id)
        else:
            raise ValueError(f"Evaluator {name} is not supported.")
