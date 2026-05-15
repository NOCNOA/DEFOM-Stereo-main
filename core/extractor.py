import importlib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import timm
from depth_anything_v2.dpt import DepthAnythingV2, _make_fusion_block
from depth_anything_v2.util.blocks import _make_scratch
from core.submodules import BasicConv, Conv2x_IN

class Feature(nn.Module):
    def __init__(self, args, out_dim):
        super(Feature, self).__init__()
        self.args = args
        model = timm.create_model('edgenext_small', pretrained=False, checkpoint_path='checkpoints/pytorch_model.bin', features_only=False)
        self.stem = model.stem
        self.stages = model.stages
        chans = [48, 96, 160, 304]
        self.chans = chans
        self.deconv32_16 = Conv2x_IN(chans[3], chans[2], deconv=True, concat=True)
        self.deconv16_8 = Conv2x_IN(chans[2]*2, chans[1], deconv=True, concat=True)
        self.deconv8_4 = Conv2x_IN(chans[1]*2, chans[0], deconv=True, concat=True)
        vit_feat_dim = 64
        self.conv4 = nn.Sequential(
          BasicConv(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, kernel_size=3, stride=1, padding=1, norm='instance'),
          ResidualBlock(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, norm_fn='instance'),
          ResidualBlock(chans[0]*2+vit_feat_dim, out_dim, norm_fn='instance'),
        )

        #self.patch_size = 14
        self.d_out = [chans[0]*2+vit_feat_dim, chans[1]*2, chans[2]*2, chans[3]] #96+128 = 224

    def forward(self, x, vit_feat):
        B,C,H,W = x.shape
        B = B//2
        x = self.stem(x)
        x4 = self.stages[0](x)
        x8 = self.stages[1](x4)
        x16 = self.stages[2](x8)
        x32 = self.stages[3](x16)

        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)
        x4 = self.deconv8_4(x8, x4)
        #print("asd", x4.shape, vit_feat.shape)
        x4 = torch.cat([x4, vit_feat], dim=1)
        x4 = self.conv4(x4)
        return [x4[:B], x8[:B], x16[:B], x32[:B]], [x4[B:], x8[B:], x16[B:], x32[B:]]

class ConvBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

    def forward(self, x):

        return self.relu(self.norm1(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1, ratio=4):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes // ratio, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes // ratio, planes // ratio, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // ratio, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // ratio)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // ratio)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes // ratio)
            self.norm2 = nn.BatchNorm2d(planes // ratio)
            self.norm3 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes // ratio)
            self.norm2 = nn.InstanceNorm2d(planes // ratio)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm4 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, d_dim, output_dim=128, norm_fn='batch', downsample=3):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))

        # depth feat convolution
        self.convd = ConvBlock(d_dim, 128, self.norm_fn)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dfeats):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        is_list = isinstance(dfeats, tuple) or isinstance(dfeats, list)
        if is_list:
            batch_dim = dfeats[0].shape[0]
            dfeats = torch.cat(dfeats, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x + self.convd(dfeats)

        x = self.conv2(x)

        if is_list:
            x = x.split(split_size=batch_dim, dim=0)

        return x


class MultiBasicEncoder(nn.Module):
    def __init__(self, d_dim, output_dim=[128, 128, 128], norm_fn='batch', downsample=3, drop_path_rate=0.2):
        super(MultiBasicEncoder, self).__init__()
        self.d_dim = d_dim
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        self.drop_path = DropPath(drop_path_rate)

        self.conv08 = ConvBlock(d_dim, 128, self.norm_fn)
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        self.conv16 = ConvBlock(d_dim, 128, self.norm_fn)
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        self.conv32 = ConvBlock(d_dim, 128, self.norm_fn)
        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs32 = nn.ModuleList(output_list)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, d_feats, num_layers=3):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        feat = x + self.drop_path(self.conv08(d_feats[0]))
        outputs08 = [f(feat) for f in self.outputs08]
        if num_layers == 1:
            return (outputs08,)

        y = self.layer4(x)
        feat = y + self.drop_path(self.conv16(d_feats[1]))
        outputs16 = [f(feat) for f in self.outputs16]

        if num_layers == 2:
            return (outputs08, outputs16)

        z = self.layer5(y)
        feat = z + self.drop_path(self.conv32(d_feats[2]))
        outputs32 = [f(feat) for f in self.outputs32]

        return (outputs08, outputs16, outputs32)


class SpatialDPTFeat(nn.Module):
    def __init__(self, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024]):
        super(SpatialDPTFeat, self).__init__()

        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=1, stride=1, padding=0)
            for out_channel in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

    def forward(self, spatial_features, out_h, out_w):
        bs = spatial_features[0].shape[0]
        out = []
        for i, x in enumerate(spatial_features):
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        layer_1_rn = F.interpolate(layer_1_rn, (out_h, out_w), mode="bilinear", align_corners=True)
        layer_2_rn = F.interpolate(layer_2_rn, (out_h // 2, out_w // 2), mode="bilinear", align_corners=True)
        layer_3_rn = F.interpolate(layer_3_rn, (out_h // 4, out_w // 4), mode="bilinear", align_corners=True)
        layer_4_rn = F.interpolate(layer_4_rn, (out_h // 8, out_w // 8), mode="bilinear", align_corners=True)

        out_features = [layer_1_rn[:bs // 2], layer_2_rn[:bs // 2], layer_3_rn[:bs // 2]]

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        return out_features, path_1[:bs // 2], path_1[bs // 2:]


class DepthAnythingV3(nn.Module):
    MODEL_CONFIGS = {
        'da3s': {
            'model_name': 'da3-small',
            'features': 64,
            'out_channels': [48, 96, 192, 384],
            'out_layers': [5, 7, 9, 11],
            'dim_in': 384,
        },
        'da3b': {
            'model_name': 'da3-base',
            'features': 128,
            'out_channels': [96, 192, 384, 768],
            'out_layers': [5, 7, 9, 11],
            'dim_in': 768,
        },
        'da3l': {
            'model_name': 'da3-large',
            'features': 256,
            'out_channels': [256, 512, 1024, 1024],
            'out_layers': [11, 15, 19, 23],
            'dim_in': 1024,
        },
    }

    def __init__(self, model_key, pretrained=True, freeze=True):
        super(DepthAnythingV3, self).__init__()

        if model_key not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported Depth Anything 3 model: {model_key}")

        self.model_cfg = self.MODEL_CONFIGS[model_key]
        self.depth_feat = SpatialDPTFeat(
            self.model_cfg['dim_in'],
            self.model_cfg['features'],
            out_channels=self.model_cfg['out_channels'],
        )
        self.depth_anything = self._build_model(pretrained)

        if freeze:
            for param in self.depth_anything.model.parameters():
                param.requires_grad = False

        self.out_dim = self.model_cfg['features']

    def _build_model(self, pretrained):
        try:
            da3_api = importlib.import_module('depth_anything_3.api')
        except ImportError as exc:
            raise ImportError(
                "Depth Anything 3 support requires the `depth_anything_3` package. "
                "Install the official repository and provide a local model directory."
            ) from exc

        if not pretrained:
            return da3_api.DepthAnything3(model_name=self.model_cfg['model_name'])

        model_source = self._resolve_model_source()
        if model_source is None:
            raise FileNotFoundError(
                "Depth Anything 3 was requested but no local model directory was found. "
                "Set `DEPTH_ANYTHING_3_MODEL_DIR` or place the model under "
                f"`checkpoints/{self.model_cfg['model_name']}`."
            )

        return da3_api.DepthAnything3.from_pretrained(model_source)

    def _resolve_model_source(self):
        env_key = f"DEPTH_ANYTHING_3_{self.model_cfg['model_name'].upper().replace('-', '_')}_DIR"
        candidates = [
            os.environ.get(env_key),
            os.environ.get('DEPTH_ANYTHING_3_MODEL_DIR'),
            os.path.join('checkpoints', self.model_cfg['model_name']),
            os.path.join('checkpoints', self.model_cfg['model_name'].replace('-', '_')),
        ]

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        return None

    def forward(self, x, out_h, out_w):
        if x.shape[0] % 2 != 0:
            raise ValueError("Depth Anything 3 expects paired stereo inputs.")

        bs = x.shape[0] // 2
        stereo_pairs = torch.stack((x[:bs], x[bs:]), dim=1)
        output = self.depth_anything(stereo_pairs, export_feat_layers=self.model_cfg['out_layers'])

        spatial_features = []
        for layer_idx in self.model_cfg['out_layers']:
            layer_feat = output['aux'][f'feat_layer_{layer_idx}']
            spatial_features.append(layer_feat.reshape(bs * 2, layer_feat.shape[2], layer_feat.shape[3], layer_feat.shape[4]))

        d_features, left_feat, right_feat = self.depth_feat(spatial_features, out_h, out_w)
        depth = output['depth'][:, 0].unsqueeze(1)
        depth = F.interpolate(depth, (out_h, out_w), mode='bilinear', align_corners=True)
        idepth = torch.reciprocal(depth.clamp_min(1e-6))

        return d_features, left_feat, right_feat, idepth


class DefomEncoder(nn.Module):
    def __init__(self, dinov2_encoder, pretrained=True, freeze=True, idepth_scale=0.25):
        super(DefomEncoder, self).__init__()
        self.dinov2_encoder = dinov2_encoder
        self.idepth_scale = idepth_scale
        self.pretrained = pretrained
        self.freeze = freeze

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
            'da3s': {'features': 64},
            'da3b': {'features': 128},
            'da3l': {'features': 256},
        }

        if self.dinov2_encoder.startswith('da3'):
            self.depth_anything = DepthAnythingV3(self.dinov2_encoder, pretrained=pretrained, freeze=freeze)
        else:
            self.depth_anything = DepthAnythingV2(**model_configs[self.dinov2_encoder])

            if pretrained and os.path.exists(f'./checkpoints/depth_anything_v2_{dinov2_encoder}.pth'):
                self.depth_anything.load_state_dict(
                    torch.load(f'./checkpoints/depth_anything_v2_{dinov2_encoder}.pth', map_location='cpu'), strict=False)
            if freeze:
                for param in self.depth_anything.pretrained.parameters():
                    param.requires_grad = False
                for param in self.depth_anything.depth_head.parameters():
                    param.requires_grad = False
        
        self.out_dim = model_configs[self.dinov2_encoder]['features']

    def forward(self, x, danv2_io_sizes):

        x = torch.cat(x, dim=0)
        ih, iw, oh, ow = danv2_io_sizes
        x = F.interpolate(x, (ih, iw), mode="bilinear", align_corners=True)

        features, left_feat, right_feat, idepth = self.depth_anything(x, oh, ow)

        bs = idepth.shape[0]
        max_idepth, _ = torch.max(idepth.view(bs, -1), dim=1)
        max_idepth = max_idepth.detach().view(bs, 1, 1, 1) + 1e-8
        idepth = idepth / max_idepth * self.idepth_scale * ow + 0.01

        return features, left_feat, right_feat, idepth
