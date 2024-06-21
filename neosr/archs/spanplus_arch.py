import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_, constant_

from neosr.utils.registry import ARCH_REGISTRY
from neosr.archs.arch_util import net_opt

upscale, __ = net_opt()


def normal_init(module, mean=0, std=1.0, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels: int, out_ch: int, scale: int = 2, groups: int = 4):
        super().__init__()

        assert in_channels >= groups and in_channels % groups == 0
        out_channels = 2 * groups * scale ** 2

        self.scale = scale
        self.groups = groups
        self.end_conv = nn.Conv2d(in_channels, out_ch, kernel_size=1)
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.register_buffer('init_pos', self._init_pos())

        normal_init(self.end_conv, std=0.001)
        normal_init(self.offset, std=0.001)
        constant_init(self.scope, val=0.)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (torch.stack(torch.meshgrid([h, h], indexing="ij")).transpose(1, 2)
                .repeat(1, self.groups, 1).reshape(1, -1, 1, 1))

    def forward(self, x):
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij")
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return self.end_conv((F.grid_sample(x.reshape(B * self.groups, -1, H, W),
                                            coords, mode='bilinear',
                                            align_corners=False,
                                            padding_mode="border")
                              .view(B, -1, self.scale * H, self.scale * W)))


class Conv3XC(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 gain: int = 1,
                 s: int = 1,
                 bias: bool = True):
        super(Conv3XC, self).__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s

        self.sk = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_in * gain, kernel_size=1, padding=0, bias=bias),
            nn.Conv2d(in_channels=c_in * gain, out_channels=c_out * gain, kernel_size=3, stride=s, padding=0,
                      bias=bias),
            nn.Conv2d(in_channels=c_out * gain, out_channels=c_out, kernel_size=1, padding=0, bias=bias),
        )
        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)

        if self.training is False:
            self.eval_conv.weight.requires_grad = False
            self.eval_conv.bias.requires_grad = False
            self.update_params()

    def update_params(self):
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1,
                                                                                                                    0,
                                                                                                                    2,
                                                                                                                    3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat

    def forward(self, x):
        if self.training:
            x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)

        return out


class SPAB(nn.Module):
    def __init__(self, in_channels: int, end: bool = False):
        super(SPAB, self).__init__()

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, in_channels, gain=2, s=1)
        self.c2_r = Conv3XC(in_channels, in_channels, gain=2, s=1)
        self.c3_r = Conv3XC(in_channels, in_channels, gain=2, s=1)
        self.act1 = nn.Mish(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.end = end

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)

        sim_att = self.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        if self.end:
            return out, out1
        return out


class SPABS(nn.Module):
    def __init__(self, feature_channels: int, n_blocks: int = 4):
        super(SPABS, self).__init__()
        self.block_1 = SPAB(feature_channels)

        self.block_n = nn.Sequential(
            *[SPAB(feature_channels)
              for _ in range(n_blocks)]
        )
        self.block_end = SPAB(feature_channels, True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain=2, s=1)
        self.conv_cat = nn.Conv2d(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        normal_init(self.conv_cat, std=0.02)

    def forward(self, x):
        out_b1 = self.block_1(x)
        out_x = self.block_n(out_b1)
        out_end, out_x_2 = self.block_end(out_x)
        out_end = self.conv_2(out_end)
        return self.conv_cat(torch.cat([x, out_end, out_b1, out_x_2], 1))


@ARCH_REGISTRY.register()
class spanplus(nn.Module):
    """
    Swift Parameter-free Attention Network for Efficient Super-Resolution
    """

    def __init__(self,
                 num_in_ch: int = 3,
                 num_out_ch: int = 3,
                 blocks: int = [4],
                 feature_channels: int = 48,
                 upscale: int = upscale,
                 drop_rate: float = 0.0,
                 upsampler: str = "dys"  # "lp", "ps"
                 ):
        super(spanplus, self).__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch if upsampler == "dys" else num_in_ch
        if not isinstance(blocks, list):
            blocks = [int(blocks)]
        if not self.training:
            drop_rate = 0
        self.feats = nn.Sequential(
            *[Conv3XC(in_channels, feature_channels, gain=2, s=1)]
             + [
                 SPABS(feature_channels, n_blocks)
                 for n_blocks in blocks
             ]
             + [nn.Dropout2d(drop_rate)])
        if upsampler == "ps":
            self.upsampler = nn.Sequential(
                nn.Conv2d(feature_channels,
                          out_channels * (upscale ** 2),
                          3, padding=1),
                nn.PixelShuffle(upscale)
            )
        else:
            self.upsampler = DySample(feature_channels, out_channels, upscale)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.feats(x)
        return self.upsampler(out)


@ARCH_REGISTRY.register()
def spanplus_xl(**kwargs):
    return spanplus(blocks=[4, 4, 4], feature_channels=96, **kwargs)


@ARCH_REGISTRY.register()
def spanplus_s(**kwargs):
    return spanplus(blocks=[2], feature_channels=32, **kwargs)


@ARCH_REGISTRY.register()
def spanplus_st(**kwargs):
    return spanplus(upsampler="ps", **kwargs)
