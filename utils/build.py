import warnings
from pathlib import Path
import torch
import torchvision.models as models


def build_model(cfg):
    name = cfg.model.lower()
    pretrained = cfg.get("pretrained", False)

    # torch model
    if hasattr(models, name):
        model_class = getattr(models, name)

        if isinstance(pretrained, bool):
            weights = "DEFAULT" if pretrained else None
        elif isinstance(pretrained, str):
            weights = None if Path(pretrained).is_file() else pretrained
        else:
            warnings.warn(f"pretrained={pretrained} is not supported, use weights='DEFAULT' instead")
            weights = "DEFAULT"

        model = model_class(weights=weights)

        if isinstance(pretrained, str) and Path(pretrained).is_file():
            ckpt = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(ckpt.get("state_dict", ckpt))

    # custom model
    # elif

    else:
        raise ValueError(f"Model {name} is not supported.")

    return model

