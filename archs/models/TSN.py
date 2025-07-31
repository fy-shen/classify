import torch
from torch import nn
import torchvision.models as models

from archs import register, get_torch_obj
from archs.modules.conv import TemporalShiftBlock, NonLocal3DWrapper


@register('model')
class TSN(nn.Module):
    def __init__(self, cfg):
        super(TSN, self).__init__()
        self.modality = cfg.modality
        self.num_classes = cfg.num_classes
        self.base_model_name = cfg.base_model.lower()
        self.pretrained = cfg.get("pretrained", False)
        self.consensus_type = cfg.consensus_type

        self.is_shift = cfg.is_shift
        self.num_seg = cfg.num_seg
        self.shift_div = cfg.shift_div
        self.temporal_pool = cfg.temporal_pool
        self.non_local = cfg.non_local
        self.dropout = cfg.dropout

        self.freeze_first_bn = cfg.freeze_first_bn
        self.partial_bn = cfg.partial_bn
        self.fc_lr5 = cfg.fc_lr5

        self._build_base_model()

    def _build_base_model(self):
        if "resnet" in self.base_model_name:
            obj = get_torch_obj(self.base_model_name, [models])
            weights = "DEFAULT" if self.pretrained else None
            self.base_model = obj(weights=weights)
            if self.is_shift:
                make_temporal_shift(
                    self.base_model, self.num_seg, self.shift_div, self.temporal_pool
                )

            if self.non_local:
                make_non_local(
                    self.base_model, self.num_seg
                )
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Dropout(p=self.dropout)
        else:
            raise ValueError(f"{self.base_model_name} is not supported")

        self.new_fc = nn.Linear(in_features, self.num_classes)
        nn.init.normal_(self.new_fc.weight, 0, 0.001)
        nn.init.constant_(self.new_fc.bias, 0)

    def train(self, mode=True):
        super(TSN, self).train(mode)
        # Freeze BN
        count = 0
        if self.partial_bn and mode:
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (1 if self.freeze_first_bn else 2):
                        m.eval()
                        # m.weight.requires_grad = False
                        # m.bias.requires_grad = False

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, nn.Linear):
                ps = list(m.parameters())
                lr5_weight.append(ps[0]) if self.fc_lr5 else normal_weight.append(ps[0])
                if len(ps) == 2:
                    lr10_bias.append(ps[1]) if self.fc_lr5 else normal_bias.append(ps[1])

            elif isinstance(m, nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self.partial_bn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self.partial_bn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]

    def get_state_dict(self, weight_path):
        ckpt = torch.load(weight_path, weights_only=False)
        ckpt = ckpt.get("state_dict", ckpt)
        ckpt = ckpt.get("model", ckpt)
        sd = {}
        for k, v in ckpt.items():
            k = k[len('module.'):] if k.startswith('module.') else k
            if "fc" not in k:
                sd[k] = v
        return sd

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        out = self.base_model(x)
        out = self.new_fc(out)

        if self.is_shift and self.temporal_pool:
            out = out.view((-1, self.num_seg // 2) + out.size()[1:])
        else:
            out = out.view((-1, self.num_seg) + out.size()[1:])

        if self.consensus_type == 'avg':
            out = out.mean(dim=1, keepdim=True)
        return out.squeeze(1)


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
