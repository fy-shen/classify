from torchvision import transforms

from archs import register


@register('transform')
def sifar10_mlp(cfg, is_train):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
