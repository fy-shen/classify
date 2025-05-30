import torchvision.models as models

import importlib


model = models.resnet18(weights='IMAGENET1K_V1')
print(model)