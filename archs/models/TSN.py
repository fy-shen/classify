from torch import nn
import torchvision.models as models

from archs import register
from archs.modules.conv import TemporalShiftBlock, NonLocal3DWrapper
from utils.build import get_torch_obj


@register('model')
class TSN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def _build_base_model(self):
        base_model = self.cfg.base_model.lower()
        if "resnet" in base_model:
            obj = get_torch_obj(base_model, [models])
            pretrained = self.cfg.get("pretrained", False)
            weights = "DEFAULT" if pretrained else None
            self.base_model = obj(weights=weights)
            if self.cfg.is_shift:
                make_temporal_shift(
                    self.base_model, self.cfg.num_seg, self.cfg.shift_div, self.cfg.temporal_pool
                )

            if self.cfg.non_local:
                make_non_local(
                    self.base_model, self.cfg.num_seg
                )


def make_temporal_shift(net, n_seg, n_div, temporal_pool=False):
    n_seg_list = [n_seg, n_seg // 2, n_seg // 2, n_seg // 2] if temporal_pool else [n_seg] * 4
    assert n_seg_list[-1] > 0

    n_round = 2 if len(list(net.layer3.children())) >= 23 else 1

    def change_block(layer, seg):
        blocks = list(layer.children())
        for i, b in enumerate(blocks):
            if i % n_round == 0:
                blocks[i].conv1 = TemporalShiftBlock(b.conv1, n_seg=seg, n_div=n_div)
        return nn.Sequential(*blocks)

    net.layer1 = change_block(net.layer1, n_seg_list[0])
    net.layer2 = change_block(net.layer2, n_seg_list[1])
    net.layer3 = change_block(net.layer3, n_seg_list[2])
    net.layer4 = change_block(net.layer4, n_seg_list[3])


def make_non_local(net, n_seg):
    net.layer2 = nn.Sequential(
        NonLocal3DWrapper(net.layer2[0], n_seg),
        net.layer2[1],
        NonLocal3DWrapper(net.layer2[2], n_seg),
        net.layer2[3],
    )
    net.layer3 = nn.Sequential(
        NonLocal3DWrapper(net.layer3[0], n_seg),
        net.layer3[1],
        NonLocal3DWrapper(net.layer3[2], n_seg),
        net.layer3[3],
        NonLocal3DWrapper(net.layer3[4], n_seg),
        net.layer3[5],
    )
