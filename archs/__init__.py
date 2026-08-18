import ast
import importlib
from pathlib import Path
from typing import Optional, Callable, Union


CUSTOM_MODELS = {}
CUSTOM_MODULES = {}
CUSTOM_LOSSES = {}
CUSTOM_TRANSFORMS = {}
CUSTOM_DATASETS = {}
CUSTOM_EVALUATOR = {}

CUSTOM_SET = {
    'model': CUSTOM_MODELS,
    'module': CUSTOM_MODULES,
    'loss': CUSTOM_LOSSES,
    'transform': CUSTOM_TRANSFORMS,
    'dataset': CUSTOM_DATASETS,
    'evaluator': CUSTOM_EVALUATOR
}

REGISTRY_MODULES = None


def register(kind: str, name: Optional[Union[str, Callable]] = None):
    def decorator(class_obj: Callable):
        key = (name or class_obj.__name__).lower()
        kind_key = kind.lower()
        custom_obj = CUSTOM_SET.get(kind_key, None)
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


def _literal_str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _register_call(decorator):
    if not isinstance(decorator, ast.Call):
        return None
    if not isinstance(decorator.func, ast.Name) or decorator.func.id != 'register':
        return None
    return decorator


def _build_registry_modules():
    registry_modules = {kind: {} for kind in CUSTOM_SET}
    root = Path(__file__).resolve().parent
    for path in sorted(root.rglob('*.py')):
        if path.name == '__init__.py':
            continue
        module_name = '.'.join((__name__, *path.relative_to(root).with_suffix('').parts))
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                continue
            for decorator in node.decorator_list:
                call = _register_call(decorator)
                if call is None or not call.args:
                    continue
                kind = _literal_str(call.args[0])
                if kind is None:
                    continue
                kind_key = kind.lower()
                if kind_key not in registry_modules:
                    continue
                name = _literal_str(call.args[1]) if len(call.args) > 1 else None
                key = (name or node.name).lower()
                registry_modules[kind_key][key] = module_name
    return registry_modules


def ensure_registered(kind: str, name: Optional[str]):
    if name is None:
        return None
    global REGISTRY_MODULES
    kind_key = kind.lower()
    key = name.lower()
    custom_obj = CUSTOM_SET.get(kind_key)
    if custom_obj is None:
        raise ValueError(f"Unknown registry kind: {kind}")
    if key not in custom_obj:
        if REGISTRY_MODULES is None:
            # 只扫描源码里的装饰器，不导入实现模块；命中配置项时再 import。
            REGISTRY_MODULES = _build_registry_modules()
        module_name = REGISTRY_MODULES.get(kind_key, {}).get(key)
        if module_name is not None:
            importlib.import_module(module_name)
    return custom_obj.get(key)


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
