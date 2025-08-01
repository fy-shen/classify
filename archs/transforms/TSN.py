import torch
from torchvision import transforms
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as v2F

from archs import register


class ToImageList(torch.nn.Module):
    def forward(self, imgs):
        return [v2F.to_image(img) for img in imgs]


class ToDtypeList(torch.nn.Module):
    def __init__(self, dtype, scale=True):
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def forward(self, imgs):
        return [v2F.to_dtype(img, dtype=self.dtype, scale=self.scale) for img in imgs]


class NormalizeList(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, imgs):
        return [v2F.normalize(img, mean=self.mean, std=self.std) for img in imgs]


class StackImageList(torch.nn.Module):
    def forward(self, imgs):
        return torch.stack(imgs, dim=0)


class PadToSquare(torch.nn.Module):
    def forward(self, image):
        h, w = v2F.get_dimensions(image)[1:]  # (C, H, W)
        diff = abs(h - w)
        if h < w:
            padding = [0, diff // 2, 0, diff - diff // 2]  # left, top, right, bottom
        else:
            padding = [diff // 2, 0, diff - diff // 2, 0]
        return v2F.pad(image, padding, fill=0)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PadImageList(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.h, self.w = size

    def forward(self, images):
        h, w = v2F.get_dimensions(images[0])[1:]
        h1 = (self.h - h) // 2
        w1 = (self.w - w) // 2
        padding = [w1, h1, self.w - w - w1, self.h - h - h1]
        return [v2F.pad(image, padding, fill=0) for image in images]


@register('transform')
def fight_tsm_rgb(cfg, is_train):
    if is_train:
        return v2.Compose([
            v2.Resize(cfg.input_size, antialias=True),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0),
            v2.RandomHorizontalFlip(),
            v2.CenterCrop(cfg.input_size),
            ToImageList(),
            ToDtypeList(torch.float32, scale=True),
            NormalizeList(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            StackImageList()
        ])
    else:
        return v2.Compose([
            v2.Resize(cfg.input_size, antialias=True),
            v2.CenterCrop(cfg.input_size),
            ToImageList(),
            ToDtypeList(torch.float32, scale=True),
            NormalizeList(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            StackImageList()
        ])


@register('transform')
def deadball_tsm_rgb(cfg, is_train):
    if is_train:
        return v2.Compose([
            v2.Resize(cfg.input_size, antialias=True),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0),
            v2.RandomHorizontalFlip(),
            ToImageList(),
            ToDtypeList(torch.float32, scale=True),
            NormalizeList(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            StackImageList()
        ])
    else:
        return v2.Compose([
            v2.Resize(cfg.input_size, antialias=True),
            ToImageList(),
            ToDtypeList(torch.float32, scale=True),
            NormalizeList(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            StackImageList()
        ])


@register('transform')
def deadball_posmlp_rgb(cfg, is_train):
    if is_train:
        return v2.Compose([
            v2.Resize(cfg.input_size, antialias=True),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0),
            v2.RandomHorizontalFlip(),
            ToImageList(),
            ToDtypeList(torch.float32, scale=True),
            PadImageList(cfg.pad_size),
            NormalizeList(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            StackImageList(),
        ])
    else:
        return v2.Compose([
            v2.Resize(cfg.input_size, antialias=True),
            ToImageList(),
            ToDtypeList(torch.float32, scale=True),
            PadImageList(cfg.pad_size),
            NormalizeList(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            StackImageList(),
        ])
