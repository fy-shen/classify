import torch
from torch import nn
import torchvision.models as models

from archs import register
from archs.tasks import BaseModel
from archs.modules.ops import DropPath


@register('model')
class PosMLP(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # B,T,C,H,W -> B,C,T,H,W
        x = self.model(x)
        return x


@register('module')
class PatchEmbed(nn.Module):
    def __init__(self, in_dim=3, embed_dim=768):
        super().__init__()
        self.proj1 = nn.Conv3d(in_dim, embed_dim // 2, (1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.norm1 = nn.BatchNorm3d(embed_dim // 2)
        self.act = nn.GELU()
        self.proj2 = nn.Conv3d(embed_dim // 2, embed_dim, (1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.norm2 = nn.BatchNorm3d(embed_dim)

    def forward(self, x):
        x = self.proj1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = self.norm2(x)

        x = x.permute(0, 2, 3, 4, 1)  # B,C,T,H,W -> B,T,H,W,C
        return x


@register('module')
class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv3d(in_dim, out_dim, (1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)  # B,T,H,W,C -> B,C,T,H,W
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1)  # B,C,T,H,W -> B,T,H,W,C
        x = self.norm(x)
        return x


@register('module')
class PermutatorBlock(nn.Module):
    def __init__(self, dim, gamma, win_size_s, win_size_t, drop_path, drop_rate=0., act_layer=nn.GELU):
        super().__init__()

        self.fc_t = PosMLPLevelT(win_size_t, dim, drop_rate, gamma, act_layer)
        self.fc_s = PosMLPLevelS(win_size_s, dim, drop_rate, gamma, act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.fc_s(x) + self.fc_t(x))
        return x


class PosMLPLevelT(nn.Module):
    def __init__(self, win_size, dim, drop_rate=0., gamma=8, act_layer=None):
        super().__init__()
        self.win_size = win_size
        self.encoder = PosMLPLayerT(win_size, dim, drop=drop_rate, gamma=gamma, act_layer=act_layer)

    def forward(self, x):
        b, t, h, w, c = x.shape
        x = x.reshape(b, t, h * w, c).permute(0, 2, 1, 3)
        x = self.encoder(x)
        x = x.permute(0, 2, 1, 3).reshape(b, t, h, w, c)
        return x


class PosMLPLayerT(nn.Module):
    def __init__(self, win_size, dim, chunks=2, drop=0., gamma=8, act_layer=nn.GELU):
        super().__init__()
        chunks = chunks
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.gate_unit = PosGUT(win_size, dim, chunks, gamma)
        self.act = act_layer()
        self.fc2 = nn.Linear(dim, dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc1(self.norm(x)))  # B, HW, T, C -> B, HW, T, C*2
        x = self.gate_unit(x)
        x = self.drop(self.fc2(x))
        return x


class PosGUT(nn.Module):
    def __init__(self, win_size, dim, chunks=2, gamma=16):
        super().__init__()
        self.chunks = chunks
        self.gate_dim = dim // chunks
        self.pos = LearnedPosMapT(win_size, gamma=gamma)

    def forward(self, x):
        if self.chunks == 1:
            u = x
            v = x
        else:
            x_chunks = x.chunk(2, dim=-1)
            u = x_chunks[0]
            v = x_chunks[1]
        u = self.pos(u)  # B, HW, T, C
        u = u * v
        return u


class LearnedPosMapT(nn.Module):
    def __init__(self, win_size, gamma=1):
        super().__init__()
        self.gamma = gamma
        self.win_size = win_size
        self.token_proj_n_bias = nn.Parameter(torch.zeros(1, self.win_size, 1))
        self.rel_local_init(self.win_size, register_name='window')

    def rel_local_init(self, win_size, register_name='window'):
        h = win_size
        w = win_size
        param = nn.Parameter(torch.zeros(2 * h - 1, self.gamma))
        nn.init.trunc_normal_(param, std=0.02)
        self.register_parameter(f'{register_name}_relative_position_bias_table', param)

        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        relative_coords = coords_h.unsqueeze(1) - coords_w.unsqueeze(0)
        relative_coords = relative_coords.unsqueeze(0).permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += h - 1
        relative_position_index = relative_coords
        self.register_buffer(
            f"{register_name}_relative_position_index",
            relative_position_index
        )

    def forward(self, x):
        posmap = self.window_relative_position_bias_table[self.window_relative_position_index.view(-1)].view(
            self.win_size, self.win_size, -1)
        posmap = posmap.permute(2, 0, 1).contiguous()

        _, wh, ww = posmap.shape
        g = self.gamma
        b, s, t, c = x.shape
        d = c // g

        # x = x.reshape(b, hw, t, d, g).permute(0, 1, 3, 4, 2).reshape(-1, g, t, 1)
        # win_weight = posmap.unsqueeze(0)
        # x = torch.matmul(win_weight, x) + self.token_proj_n_bias.unsqueeze(0)
        # x = x.reshape(b, hw, d, g, t).permute(0, 1, 4, 2, 3).reshape(b, hw, t, c)
        win_weight = posmap.reshape(-1, g, wh, ww).permute(0, 2, 3, 1)
        x = x.reshape(b, s, t, d, g)
        x = torch.einsum('wmns,bwnvs->bwmvs', win_weight, x) + self.token_proj_n_bias.unsqueeze(0).unsqueeze(-1)
        x = x.reshape(b, s, t, c)
        return x


class PosMLPLevelS(nn.Module):
    """ Single hierarchical level of a Nested Transformer
    """

    def __init__(self, win_size, dim, drop_rate=0., gamma=8, act_layer=None):
        super().__init__()
        self.win_size = win_size

        self.encoder = PosMLPLayerS(win_size, dim, drop=drop_rate, gamma=gamma, act_layer=act_layer)

    def forward(self, x):
        """
        expects x as (1, C, H, W)
        """
        b, t, h, w, c = x.shape
        bt = b * t
        x = x.reshape(bt, h, w, c)
        x = x.reshape(bt, h // self.win_size, self.win_size, w // self.win_size, self.win_size, c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(bt, -1, self.win_size ** 2, c)  # BT, WIN_NUM, WIN_SIZE, C

        x = self.encoder(x)

        x = x.reshape(bt, -1, self.win_size, self.win_size, c)
        x = x.reshape(bt, h // self.win_size, w // self.win_size, self.win_size, self.win_size, c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(bt, h, w, c)
        x = x.reshape(b, t, h, w, c)
        return x


class PosMLPLayerS(nn.Module):
    def __init__(self, win_size, dim, chunks=2, drop=0., gamma=8, act_layer=nn.GELU):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.act = act_layer()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.gate_unit = PosGUTS(win_size, dim, chunks, gamma)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None):
        x = self.act(self.fc1(self.norm(x)))
        x = self.gate_unit(x)
        x = self.drop(self.fc2(x))
        return x


class PosGUTS(nn.Module):
    def __init__(self, win_size, dim, chunks=2, gamma=16, quadratic=True, pos_only=True):
        super().__init__()
        self.chunks = chunks
        self.gate_dim = dim // chunks
        self.seq_len = win_size * win_size
        self.pos = LearnedPosMapS(win_size, gamma=gamma)
        self.quadratic = quadratic
        self.pos_only = pos_only

        if not self.pos_only:
            self.token_proj_n_weight = nn.Parameter(torch.zeros(1, self.seq_len, self.seq_len))
            nn.init.trunc_normal_(self.token_proj_n_weight, std=1e-6)

    def forward(self, x):
        if self.chunks == 1:
            u = x
            v = x
        else:
            x_chunks = x.chunk(2, dim=-1)
            u = x_chunks[0]
            v = x_chunks[1]
        if not self.pos_only and not self.quadratic:
            u = self.pos(u, self.token_proj_n_weight)
        else:
            u = self.pos(u)

        u = u * v
        return u


class LearnedPosMapS(nn.Module):
    def __init__(self, win_size, gamma=1):
        super().__init__()
        self.gamma = gamma
        self.win_size = win_size
        self.seq_len = win_size * win_size
        self.token_proj_n_bias = nn.Parameter(torch.zeros(1, self.seq_len, 1))
        self.rel_local_init(self.win_size, register_name='window')

    def rel_local_init(self, win_size, register_name='window'):
        h = win_size
        w = win_size
        parm = nn.Parameter(torch.zeros((2 * h - 1) * (2 * w - 1), self.gamma))
        nn.init.trunc_normal_(parm, std=0.02)
        self.register_parameter(
            f'{register_name}_relative_position_bias_table',
            nn.Parameter(torch.zeros((2 * h - 1) * (2 * w - 1), self.gamma))
        )

        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += h - 1
        relative_coords[:, :, 1] += w - 1
        relative_coords[:, :, 0] *= 2 * w - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer(
            f"{register_name}_relative_position_index",
            relative_position_index
        )

    def forward(self, x, weight=None):
        posmap = self.window_relative_position_bias_table[self.window_relative_position_index.view(-1)].view(
            self.seq_len, self.seq_len, -1)
        posmap = posmap.permute(2, 0, 1).contiguous()
        _, wh, ww = posmap.shape
        g = self.gamma
        b, n, s, c = x.shape
        d = c // g

        win_weight = posmap + weight if weight is not None else posmap
        # x = x.reshape(b, n, s, d, g).permute(0, 1, 3, 4, 2).reshape(-1, g, s, 1)
        # win_weight = win_weight.unsqueeze(0)
        # x = torch.matmul(win_weight, x) + self.token_proj_n_bias.unsqueeze(0)
        # x = x.reshape(b, n, d, g, s).permute(0, 1, 4, 2, 3).reshape(b, n, s, c)
        win_weight = win_weight.reshape(-1, g, wh, ww).permute(0, 2, 3, 1)
        x = x.reshape(b, n, s, d, g)
        x = torch.einsum('wmns,bwnvs->bwmvs', win_weight, x) + self.token_proj_n_bias.unsqueeze(0).unsqueeze(-1)
        x = x.reshape(b, n, s, c)
        return x
