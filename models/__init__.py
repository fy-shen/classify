from typing import Optional, Callable, Type, Union
import torch.nn as nn


CUSTOM_MODELS = {}


def register_model(name: Optional[Union[str, Callable]] = None):
    def decorator(model_class: Type[nn.Module]):
        key = name or model_class.__name__.lower()
        CUSTOM_MODELS[key] = model_class
        return model_class

    # @register_model 调用，`name` 为类对象
    if callable(name):
        model_cls = name
        name = None
        return decorator(model_cls)

    # @register_model(name) 调用，`name` 为字符串，即模型名称
    return decorator
