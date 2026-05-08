import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update_sele import BasicSelectiveMultiUpdateBlock ,ScaleSelectiveMultiUpdateBlock, GlobalStereoCorrectionBlock
from core.extractor import MultiBasicEncoder, DefomEncoder, Feature
from core.corr import CorrBlock1D2
from core.utils.utils import coords_grid, upflow, get_danv2_io_size
from core.submodules import build_gwc_volume, build_concat_volume, ResnetBasicBlock3D, BasicConv, FeatureAtt, CostVolumeDisparityAttention, ChannelAttentionEnhancement, SpatialAttentionExtractor
from core.galerkin import simple_attn_3d3, simple_attn_3d_hw_patch
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

# class hourglass(nn.Module):
#     def __init__(self, cfg, in_channels, feat_dims=None):
#         super().__init__()
#         self.cfg = cfg
#         self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
#                                              padding=1, stride=2, dilation=1),
#                                    Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))

#         self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
#                                              padding=1, stride=2, dilation=1),
#                                    Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17))

#         self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
#                                              padding=1, stride=2, dilation=1),
#                                    Conv3dNormActReduced(in_channels*6, in_channels*6, kernel_size=3, kernel_disp=17))


#         self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
#                                   relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        
#         self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
#                                   relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

#         self.conv1_up = BasicConv(in_channels*2, in_channels, deconv=True, is_3d=True, bn=True,
#                                   relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
#         self.conv_out = nn.Sequential(
#           Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
#           Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
#         )

#         self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
#                                    Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),
#                                    Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),)

#         self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
#                                    Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17),
#                                    Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))
#         self.atts = nn.ModuleDict({
#           "4": CostVolumeDisparityAttention(d_model=in_channels, nhead=4, dim_feedforward=in_channels, norm_first=False, num_transformer=4, max_len=self.cfg.max_disp // 16),
#         })
#         self.conv_patch = nn.Sequential(
#           nn.Conv3d(in_channels, in_channels, kernel_size=4, stride=4, padding=0, groups=in_channels),
#           nn.BatchNorm3d(in_channels),
#         )

#         self.feature_att_8 = FeatureAtt(in_channels*2, feat_dims[1])
#         self.feature_att_16 = FeatureAtt(in_channels*4, feat_dims[2])
#         self.feature_att_32 = FeatureAtt(in_channels*6, feat_dims[3])
#         self.feature_att_up_16 = FeatureAtt(in_channels*4, feat_dims[2])
#         self.feature_att_up_8 = FeatureAtt(in_channels*2, feat_dims[1])

#     def forward(self, x, features):
#         conv1 = self.conv1(x)
#         conv1 = self.feature_att_8(conv1, features[1])

#         conv2 = self.conv2(conv1)
#         conv2 = self.feature_att_16(conv2, features[2])

#         conv3 = self.conv3(conv2)
#         conv3 = self.feature_att_32(conv3, features[3])

#         conv3_up = self.conv3_up(conv3)
#         #print("ddffss", conv3_up.shape, conv2.shape)
#         conv2 = torch.cat((conv3_up, conv2), dim=1)
#         conv2 = self.agg_0(conv2)
#         conv2 = self.feature_att_up_16(conv2, features[2])

#         conv2_up = self.conv2_up(conv2)
#         conv1 = torch.cat((conv2_up, conv1), dim=1)
#         conv1 = self.agg_1(conv1)
#         conv1 = self.feature_att_up_8(conv1, features[1])

#         conv = self.conv1_up(conv1)
#         x = self.conv_patch(x)
#         x = self.atts["4"](x)
#         x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=False)
#         conv = conv + x
#         conv = self.conv_out(conv)

#         return conv

class DEFOMStereo(nn.Module):
    def __init__(self, args):
        super(DEFOMStereo, self).__init__()
        self.args = args
        volume_dim = 16 #这个volumedim实际上按需设置即可
        self.register_buffer('mean', torch.tensor([[0.485, 0.456, 0.406]])[..., None, None] * 255)
        self.register_buffer('std', torch.tensor([[0.229, 0.224, 0.225]])[..., None, None] * 255)

        self.defomencoder = DefomEncoder(args.dinov2_encoder, idepth_scale=args.idepth_scale)
        self.low_channel = nn.Conv2d(128, 12, kernel_size=1, padding=0)
        context_dims = args.hidden_dims
        self.fnet2 = Feature(self.args, out_dim=128)
        #self.fnet = BasicEncoder(self.defomencoder.out_dim, output_dim=128, norm_fn='instance', downsample=args.n_downsample)
        self.max_disp = self.args.max_disp
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.update_block = BasicSelectiveMultiUpdateBlock(self.args, hidden_dim=args.hidden_dims[0], volume=volume_dim+1)
        self.scale_update_block = ScaleSelectiveMultiUpdateBlock(self.args, hidden_dim=args.hidden_dims[0], volume=volume_dim+1)
        #self.confidence_conv = nn.Conv3d(32, 1, kernel_size=1, bias=True)
        self.cnet = MultiBasicEncoder(self.defomencoder.out_dim, output_dim=[args.hidden_dims, context_dims],
                                      norm_fn=args.context_norm, downsample=args.n_downsample)
        self.corr_stem = nn.Sequential(
            nn.Conv3d(32, volume_dim, kernel_size=1),
            BasicConv(volume_dim, volume_dim, kernel_size=3, padding=1, is_3d=True),
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
            )
        self.sam = SpatialAttentionExtractor()
        self.cam = ChannelAttentionEnhancement(self.args.hidden_dims[0])
        #self.corr_feature_att = FeatureAtt(volume_dim, 128)
        #self.cost_agg = hourglass(cfg=self.args, in_channels=volume_dim, feat_dims=self.fnet2.d_out)
        self.disp_att = CostVolumeDisparityAttention(d_model=volume_dim, nhead=2, dim_feedforward=volume_dim, norm_first=False, num_transformer=4, max_len=self.args.max_disp)
        #self.galerkin_attH = simple_attn_3d4(volume_dim, 4)
        self.galerkin_att_w = simple_attn_3d3(volume_dim, 2)
        self.galerkin_att_p = simple_attn_3d_hw_patch(volume_dim, 2)
        #self.galerkin_att = CostVolumeDisparityAttention2(d_model=volume_dim, nhead=4, dim_feedforward=volume_dim, norm_first=False, num_transformer=4, max_len=self.args.max_disp//4, resize_embed= True)
        # self.conv_out = nn.Sequential(
        #   Conv3dNormActReduced(volume_dim, volume_dim, kernel_size=3, kernel_disp=17),
        #   Conv3dNormActReduced(volume_dim, volume_dim, kernel_size=3, kernel_disp=17),
        # )

        self.global_corr_block = GlobalStereoCorrectionBlock(
            self.args,
            hidden_dim=args.hidden_dims[0],
            context_dim=context_dims[0],
            volume_dim=volume_dim,
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv3d(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(volume_dim, volume_dim),  #待定为batchnorm
            nn.ReLU() # 或 ReLU
        )
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_coords(self, img):
        """ Disparity is represented as difference between two vertical coordinate grids disp
            = coords0[:, :1] - coords1[:, :1] """
        N, _, H, W = img.shape

        coords = coords_grid(N, H, W)[:, :1].to(img.device)

        return coords

    def upsample_flow(self, flow, mask):
        """ Upsample disparity field [H/scale, W/scale, 1] -> [H, W, 1] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor * H, factor * W)

    def forward(self, image1, image2, iters=12, scale_iters=3, test_mode=False):
        """ Estimate optical flow between pair of frames """
        b, c, h, w = image1.shape
        #step = ((self.args.max_disp) / ((self.args.max_disp//4) * scale)).float()
        #print(f"stepqweqweqwe: {step, scale, w}", flush = True)
        image1 = ((image1 - self.mean)/self.std).contiguous().float()
        image2 = ((image2 - self.mean)/self.std).contiguous().float()

        b, _, h, w = image1.shape
        danv2_io_sizes = get_danv2_io_size(h, w, self.args.n_downsample)

        # run the context network
        d_features, dfeat1, dfeat2, disp = self.defomencoder([image1, image2], danv2_io_sizes)
        disp = disp.float()  # Ensure disp is float32
        fmapl, fmapr = self.fnet2(torch.cat([image1, image2], dim = 0), torch.cat([dfeat1, dfeat2], dim = 0))
        with autocast(enabled=self.args.mixed_precision):
            # if(torch.isnan(dfeat1).any() and torch.isnan(dfeat2).any()): 
            #     print(1111);


            #fmap1, fmap2 = self.fnet([image1, image2], [dfeat1, dfeat2])
            cnet_list = self.cnet(image1, d_features)
            #print("aaddss", fmapl[0].shape, fmapl[1].shape, fmapl[2].shape, fmapl[3].shape)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning
            inp_list = [self.cam(x) * x for x in inp_list]
            att = [self.sam(x) for x in inp_list]
            coords = self.initialize_coords(net_list[0])

            fmap1, fmap2 = fmapl[0], fmapr[0]
            # if(torch.isnan(fmap1).any() and torch.isnan(fmap2).any()): 
            #     print(2222);
            #print(f"fmap1: {fmap1.shape}, fmap2: {fmap2.shape}", flush = True)  
            #fmap1: torch.Size([4, 256, 80, 184]), fmap2: torch.Size([4, 256, 80, 184])
            
            gwc_volume = build_gwc_volume(fmap1, fmap2, self.max_disp, 8)
            concat_volume = build_concat_volume(fmap1, fmap2, self.max_disp, self.low_channel)
            #print(f"gwc_volume: {gwc_volume.shape}, corr_volume: {corr_volume.shape}, concat_volume: {concat_volume.shape}", flush=True)
            #gwc_volume: torch.Size([4, 8, 184, 80, 184]), corr_volume: torch.Size([4, 1, 184, 80, 184]), concat_volume: torch.Size([4, 24, 184, 80, 184])
            volume = torch.cat([gwc_volume, concat_volume], dim = 1)  #B, 32=(8+12*2), H, W
            #print(f"volume: {volume.shape}", flush=True)
            #volume: torch.Size([1, 32, 184, 80, 184])
            #del gwc_volume, concat_volume
            # if(torch.isnan(volume).any()): 
            #     print(2222);
            volume = self.corr_stem(volume)
            #torch.save(volume, "visual/volume1.pth")
            #volume = self.corr_feature_att(volume, fmap1)
            #volume = self.cost_agg(volume, fmapl)#[16,140,80,140]  [[128,80,140],[],[],[]]
            #volume_gaH = self.galerkin_attH(volume)
            volume_gaW = self.galerkin_att_w(volume)
            volume_gaP = self.galerkin_att_p(volume)
            #torch.save(volume_gaW, "visual/volume2.pth")
            volume_dt = self.disp_att(volume)
            # if(torch.isnan(volume_gaW).any() or torch.isnan(volume_gaP).any() or torch.isnan(volume_dt).any()): 
            #     print(3333);
            #torch.save(volume_dt, "visual/volume3.pth")
            volume = volume_dt + volume_gaW + volume_gaP
            volume = self.fusion_conv(volume).float()
            #print(f"volume: {volume.shape}", flush=True)
            #volume: torch.Size([1, 16, 184, 80, 184])
            #del volume_ga


        corr_fn = CorrBlock1D2(volume, coords, fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels,
                              scale_list=self.args.scale_list, scale_corr_radius=self.args.scale_corr_radius)

        disp_predictions = []
        gc_aux = None
        global_corr_applied = False
        for itr in range(iters):
            disp = disp.detach()

            if itr < scale_iters:
                corr = corr_fn(disp, scaling=True)  # index correlation volume
                with autocast(enabled=self.args.mixed_precision):
                    net_list, up_mask, scale_disp = self.scale_update_block(net_list, inp_list, corr, disp, att)
                # F(t+1) = \Scale(t) x F(t)
                disp = scale_disp * disp
            else:
                if (not global_corr_applied):
                    corr_gc = corr_fn(disp, scaling=False)  # 当前 disp 对应的 local cue
                    with autocast(enabled=self.args.mixed_precision):
                        disp, gc_aux = self.global_corr_block(
                            volume=volume,
                            disp=disp,
                            corr=corr_gc,
                            net=net_list[0],
                            inp=inp_list[0],
                        )
                    global_corr_applied = True
                corr = corr_fn(disp, scaling=False)  # index correlation volume
                with autocast(enabled=self.args.mixed_precision):
                    net_list, up_mask, delta_disp = self.update_block(net_list, inp_list, corr, disp, att)
                    # To avoid unstability, we limit the disparity update within the searching range.
                    delta_disp = torch.clip(delta_disp, min=-2**(self.args.corr_levels-1)*self.args.corr_radius,
                                            max=2**(self.args.corr_levels-1)*self.args.corr_radius)
                    delta_disp = delta_disp
                # F(t+1) = F(t) + \Delta(t)
                disp = disp + delta_disp

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters - 1:
                continue

            # upsample predictions
            if up_mask is None:
                disp_up = upflow(disp, factor=2 ** self.n_downsample)
            else:
                #print(f"up_mask: {up_mask.shape}", flush=True)
                #print(f"disp: {disp.shape}", flush=True)
                disp_up = self.upsample_flow(disp, up_mask)
            # if(torch.isnan(disp_up).any()): 
            #     print(2222,"asdasd ",itr);
            disp_predictions.append(disp_up)

        if test_mode:
            return disp_up

        return disp_predictions


