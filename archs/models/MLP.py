import torch.nn as nn

from archs import register
from archs.tasks import Model


@register('model')
class MLP(Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)
