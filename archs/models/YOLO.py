import torch.nn as nn

from archs import register
from archs.tasks import BaseModel


@register('model')
class YOLO(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x = x.squeeze(1)
        return self.predict(x)
