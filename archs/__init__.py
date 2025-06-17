import os
import importlib
import pkgutil
from typing import Optional, Callable, Union


CUSTOM_MODELS = {}
CUSTOM_MODULES = {}
CUSTOM_LOSSES = {}
CUSTOM_TRANSFORMS = {}
CUSTOM_DATASETS = {}

CUSTOM_SET = {
    'model': CUSTOM_MODELS,
    'module': CUSTOM_MODULES,
    'loss': CUSTOM_LOSSES,
    'transform': CUSTOM_TRANSFORMS,
    'dataset': CUSTOM_DATASETS,
}


def register(kind: str, name: Optional[Union[str, Callable]] = None):
    def decorator(class_obj: Callable):
        key = name or class_obj.__name__.lower()
        custom_obj = CUSTOM_SET.get(kind, None)
        if custom_obj is None:
            raise Warning("Unknown kind '{}', register '{}' failed".format(kind, name))
        else:
            custom_obj[key] = class_obj
        return class_obj

    if callable(name):
        obj = name
        name = None
        return decorator(obj)
    return decorator


def get_torch_obj(name, modules):
    name_low = name.lower()
    for mod in modules:
        # 大小写完全匹配
        if hasattr(mod, name):
            return getattr(mod, name)
        # 大小写模糊匹配
        for key in dir(mod):
            if key.lower() == name_low:
                return getattr(mod, key)
    return None


for _, modname, ispkg in pkgutil.walk_packages(path=[os.path.dirname(__file__)], prefix=f"{__name__}."):
    importlib.import_module(modname)
