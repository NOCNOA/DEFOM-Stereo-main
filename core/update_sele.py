# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch,pdb,os,sys
import torch.nn as nn
import torch.nn.functional as F
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.submodules import *
from core.extractor import *

class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DispHead, self).__init__()
        self.conv = nn.Sequential(
          nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
          nn.ReLU(),
          EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),
          EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),
          nn.Conv2d(input_dim, output_dim, 3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)
        h = (1-z) * h + z * q
        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, args, cor_planes=8):
        super(BasicMotionEncoder, self).__init__()
        self.args = args
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 256, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+256, 128-1, 3, padding=1)

    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class RaftConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=3):
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, x, hx):
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h


class SelectiveConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, small_kernel_size=1, large_kernel_size=3, patch_size=None):
        super(SelectiveConvGRU, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim+hidden_dim, input_dim+hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.small_gru = RaftConvGRU(hidden_dim, input_dim, small_kernel_size)
        self.large_gru = RaftConvGRU(hidden_dim, input_dim, large_kernel_size)

    def forward(self, att, h, *x):
        x = torch.cat(x, dim=1)
        x = self.conv0(x)
        hx = torch.cat([x, h], dim=1)
        hx = self.conv1(hx)
        h = self.small_gru(h, x, hx) * att + self.large_gru(h, x, hx) * (1 - att)

        return h


class BasicSelectiveMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, volume=16):
        super().__init__()
        self.args = args
        cor_planes = args.corr_levels * (2*args.corr_radius + 1) * volume #2 * 9 * 16 = 288  -> 3 * 9 * 16 = 432
        self.encoder = BasicMotionEncoder(args, cor_planes)

        if args.n_gru_layers == 3:
            self.gru16 = SelectiveConvGRU(hidden_dim, hidden_dim * 2)
        if args.n_gru_layers >= 2:
            self.gru08 = SelectiveConvGRU(hidden_dim, hidden_dim * (args.n_gru_layers == 3) + hidden_dim * 2)
        self.gru04 = SelectiveConvGRU(hidden_dim, hidden_dim * (args.n_gru_layers > 1) + hidden_dim * 2)
        self.disp_head = DispHead(hidden_dim, 256)

        factor = 2**self.args.n_downsample

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor**2)*9, 1, padding=0))

        # self.mask1 = nn.Sequential(
        #     nn.Conv2d(128, 64, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 32, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     )

    def forward(self, net, inp, corr, disp, att):
        if self.args.n_gru_layers == 3:
            net[2] = self.gru16(att[2], net[2], inp[2], pool2x(net[1]))
        if self.args.n_gru_layers >= 2:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]))

        motion_features = self.encoder(disp, corr)
        motion_features = torch.cat([inp[0], motion_features], dim=1)
        if self.args.n_gru_layers > 1:
            net[0] = self.gru04(att[0], net[0], motion_features, interp(net[1], net[0]))

        delta_disp = self.disp_head(net[0])

        # scale mask to balence gradients
        mask = .25 * self.mask(net[0])
        return net, mask, delta_disp

class GlobalStereoCorrectionBlock(nn.Module):
    """
    SU 后、DU 前的一次性全局 stereo 校正模块
    输入:
        volume:    [B, Cv, D, H, W]   fusion_conv 后的 stereo volume
        disp:      [B, 1, H, W]       SU 后当前 disparity
        corr:      [B, Cc, H, W]      当前 disp 对应的 local lookup feature
        net:       [B, Ch, H, W]      当前 1/4 hidden state
        inp:       [B, Ci, H, W]      当前 1/4 context feature
    输出:
        disp_new:  [B, 1, H, W]
        aux:       调试/可选监督用中间结果
    """
    def __init__(self, args, hidden_dim=128, context_dim=128, volume_dim=16):
        super().__init__()
        self.args = args

        # CorrBlock1D2 的 local lookup 通道数:
        # 每个 offset 包含 volume_dim 个 volume 特征 + 1 个 corr 特征
        corr_channels = args.corr_levels * (2 * args.corr_radius + 1) * (volume_dim + 1)

        mid3d = max(32, hidden_dim // 2)
        mid2d = max(32, hidden_dim // 2)
        disp_ch = max(16, hidden_dim // 4)

        # 1) 从全 disparity 维 stereo volume 生成 absolute stereo proposal
        self.volume_head = nn.Sequential(
            nn.Conv3d(volume_dim, hidden_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, mid3d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid3d, 1, kernel_size=1, padding=0),
        )

        # 2) 压缩当前 local corr / volume lookup 特征
        self.corr_proj = nn.Sequential(
            nn.Conv2d(corr_channels, mid2d, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid2d, mid2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 3) 编码当前 disparity
        self.disp_proj = nn.Sequential(
            nn.Conv2d(1, disp_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 4) gate: 决定在每个像素上，global proposal 对当前 disp 修正多少
        # 输入: net, inp, corr_feat, disp_feat, disp_prop, conf, entropy
        gate_in_ch = hidden_dim + context_dim + mid2d + disp_ch + 3
        self.gate_head = nn.Sequential(
            nn.Conv2d(gate_in_ch, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, mid2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid2d, 1, kernel_size=3, padding=1),
        )

    def forward(self, volume, disp, corr, net, inp):
        """
        volume: [B, Cv, D, H, W]
        disp:   [B, 1, H, W]
        corr:   [B, Cc, H, W]
        net:    [B, Ch, H, W]
        inp:    [B, Ci, H, W]
        """
        # ----- stereo absolute proposal from full disparity axis -----
        logits = self.volume_head(volume).squeeze(1)          # [B, D, H, W]
        prob = torch.softmax(logits, dim=1)

        D = prob.shape[1]
        disp_values = torch.arange(
            D, device=prob.device, dtype=prob.dtype
        ).view(1, D, 1, 1)

        disp_prop = torch.sum(prob * disp_values, dim=1, keepdim=True)   # [B,1,H,W]
        conf = prob.max(dim=1, keepdim=True).values                       # [B,1,H,W]
        entropy = -(prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()).sum(
            dim=1, keepdim=True
        )                                                                 # [B,1,H,W]

        # ----- gate conditioned on current state -----
        corr_feat = self.corr_proj(corr)
        disp_feat = self.disp_proj(disp)

        gate_in = torch.cat(
            [net, inp, corr_feat, disp_feat, disp_prop, conf, entropy], dim=1
        )
        alpha = torch.sigmoid(self.gate_head(gate_in))                    # [B,1,H,W]

        # ----- gated correction -----
        disp_new = disp + alpha * (disp_prop - disp)

        aux = {
            "logits": logits,
            "prob": prob,
            "disp_prop": disp_prop,
            "alpha": alpha,
            "conf": conf,
            "entropy": entropy,
        }
        return disp_new, aux

class ScaleSelectiveMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, volume=16):
        super().__init__()
        self.args = args
        cor_planes = len(args.scale_list) * (2*args.scale_corr_radius + 1) * volume #8 * 5 * 16 = 640  3 * 9 * 16 = 432  2 * 15 * 16 = 480
        self.encoder = BasicMotionEncoder(args, cor_planes)

        if args.n_gru_layers == 3:
            self.gru16 = SelectiveConvGRU(hidden_dim, hidden_dim * 2)
        if args.n_gru_layers >= 2:
            self.gru08 = SelectiveConvGRU(hidden_dim, hidden_dim * (args.n_gru_layers == 3) + hidden_dim * 2)
        self.gru04 = SelectiveConvGRU(hidden_dim, hidden_dim * (args.n_gru_layers > 1) + hidden_dim * 2)
        self.disp_head = DispHead(hidden_dim, 256)
        factor = 2**self.args.n_downsample

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor**2)*9, 1, padding=0))
        # self.mask1 = nn.Sequential(
        #     nn.Conv2d(128, 64, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 32, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     )
    def forward(self, net, inp, corr, disp, att):
        if self.args.n_gru_layers == 3:
            net[2] = self.gru16(att[2], net[2], inp[2], pool2x(net[1]))
        if self.args.n_gru_layers >= 2:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]))

        motion_features = self.encoder(disp, corr)
        motion_features = torch.cat([inp[0], motion_features], dim=1)
        if self.args.n_gru_layers > 1:
            net[0] = self.gru04(att[0], net[0], motion_features, interp(net[1], net[0]))

        delta_disp = self.disp_head(net[0])
        scale_disp = F.relu6(torch.exp(.25*delta_disp))
        # scale mask to balence gradients
        mask = .25 * self.mask(net[0])
        return net, mask, scale_disp