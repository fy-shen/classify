from archs import register
from archs.tasks import Model


@register('model')
class MLP(Model):
    def __init__(self, cfg):
        super().__init__(cfg)


