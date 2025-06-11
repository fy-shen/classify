import os
import importlib
import pkgutil
from typing import Optional, Callable, Union


CUSTOM_MODELS = {}
CUSTOM_MODULES = {}
CUSTOM_LOSSES = {}
CUSTOM_TRANSFORMS = {}

CUSTOM_SET = {
    'model': CUSTOM_MODELS,
    'module': CUSTOM_MODULES,
    'loss': CUSTOM_LOSSES,
    'transform': CUSTOM_TRANSFORMS
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


for _, modname, ispkg in pkgutil.walk_packages(path=[os.path.dirname(__file__)], prefix=f"{__name__}."):
    importlib.import_module(modname)
