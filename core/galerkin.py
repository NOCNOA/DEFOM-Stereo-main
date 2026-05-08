import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.backends.cudnn.enabled = False

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out

class simple_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = GELU()
    
    def forward(self, x, name='0'):
        B, C, H, W = x.shape
        bias = x

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.headc) # B, H*W, heads, 3*headc
        qkv = qkv.permute(0, 2, 1, 3) # B, heads, H*W, 3*headc
        q, k, v = qkv.chunk(3, dim=-1) # B, heads, H*W, headc

        k = self.kln(k) # B, heads, H*W, headc
        v = self.vln(v) # B, heads, H*W, headc

        
        v = torch.matmul(k.transpose(-2,-1), v / (H*W)) # B, heads, H*W, H*W

        if torch.isnan(v).any() or torch.isnan(v).any():
            print('v, isnan-isinf_v', torch.isnan(x).any(), torch.isnan(q).any(), torch.isnan(k).any(), torch.isnan(v).any(), flush=True)
        
        v = torch.matmul(q, v)

        if torch.isnan(v).any() or torch.isnan(v).any():
            print('v_, isnan-isinf_v_', torch.isnan(q).any(), torch.isnan(k).any(), torch.isnan(v).any(), flush=True)

        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

        ret = v.permute(0, 3, 1, 2) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias

class simple_attn_3d(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv3d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv3d(midc, midc, 1)
        self.o_proj2 = nn.Conv3d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = GELU()
    
    def forward(self, x, name='0'):
        B, C, D, H, W = x.shape
        bias = x

        # print(f'galerkin before {x.shape}', flush=True)
        
        qkv = self.qkv_proj(x)
        qkv = qkv.permute(0, 2, 3, 4, 1)
        qkv = qkv.reshape(B*D, H*W, self.heads, 3*self.headc)

        # print(f'galerkin after {qkv.shape}', flush=True)

        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1) # B*D, heads, H*W, headc

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2,-1), v/(H*W))
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, D, H, W, C)

        ret = v.permute(0, 4, 1, 2, 3) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias
    
class simple_attn_3d2(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv3d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv3d(midc, midc, 1)
        self.o_proj2 = nn.Conv3d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = GELU()
    
    def forward(self, x, name='0'):
        B, C, D, H, W = x.shape
        bias = x

        # print(f'galerkin before {x.shape}', flush=True)
        
        qkv = self.qkv_proj(x)
        qkv = qkv.permute(0, 3, 4, 2, 1)  #B H W D C
        qkv = qkv.reshape(B*H*W, D, self.heads, 3*self.headc)

        # print(f'galerkin after {qkv.shape}', flush=True)

        qkv = qkv.permute(0, 2, 1, 3)  #B*H*W, self.heads, D, 3*self.headc
        q, k, v = qkv.chunk(3, dim=-1) # B*H*W, heads, D, headc

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2,-1), v/(D))
        v = torch.matmul(q, v)   # B*H*W, heads, D, headc
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, D, C)# B*H*W, heads, D, headc  ->  B*H*W, D, heads, headc

        ret = v.permute(0, 4, 3, 1, 2) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias

class simple_attn_3d3(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv3d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv3d(midc, midc, 1)
        self.o_proj2 = nn.Conv3d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = GELU()
    
    def forward(self, x, name='0'):
        B, C, D, H, W = x.shape
        bias = x
        
        # print(f'galerkin before {x.shape}', flush=True)
        
        qkv = self.qkv_proj(x)
        qkv = qkv.permute(0, 2, 3, 4, 1)
        qkv = qkv.reshape(B*D*H, W, self.heads, 3*self.headc)

        # print(f'galerkin after {qkv.shape}', flush=True)

        qkv = qkv.permute(0, 2, 1, 3)  #B*D*H, self.heads, W, 3*self.headc
        q, k, v = qkv.chunk(3, dim=-1) #H*B*D, heads, W, headc

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2,-1), v/(W)) #H*B*D, heads, headc, headc
        v = torch.matmul(q, v)  #B*D*H, heads, W, headc
        v = v.permute(0, 2, 1, 3).reshape(B, D, H, W, C) #B, D, H, W, C

        ret = v.permute(0, 4, 1, 2, 3) + bias #B, C, D, H, W
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias 
        
        return bias

class simple_attn_3d3(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv3d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv3d(midc, midc, 1)
        self.o_proj2 = nn.Conv3d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = GELU()
    
    def forward(self, x, name='0'):
        B, C, D, H, W = x.shape
        bias = x
        
        # print(f'galerkin before {x.shape}', flush=True)
        
        qkv = self.qkv_proj(x)
        qkv = qkv.permute(0, 2, 3, 4, 1)
        qkv = qkv.reshape(B*D*H, W, self.heads, 3*self.headc)

        # print(f'galerkin after {qkv.shape}', flush=True)

        qkv = qkv.permute(0, 2, 1, 3)  #B*D*H, self.heads, W, 3*self.headc
        q, k, v = qkv.chunk(3, dim=-1) #H*B*D, heads, W, headc

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2,-1), v/(W)) #H*B*D, heads, headc, headc
        v = torch.matmul(q, v)  #B*D*H, heads, W, headc
        v = v.permute(0, 2, 1, 3).reshape(B, D, H, W, C) #B, D, H, W, C

        ret = v.permute(0, 4, 1, 2, 3) + bias #B, C, D, H, W
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias 
        
        return bias

class simple_attn_3d4(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv3d(3*midc, 3*midc, 1)
        self.o_proj1 = nn.Conv3d(midc, midc, 1)
        self.o_proj2 = nn.Conv3d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = GELU()
    
    def forward(self, x, name='0'):
        B, C, D, H, W = x.shape
        bias = x
        x = torch.cat([x, x, x], dim=1)
        # print(f'galerkin before {x.shape}', flush=True)
        
        qkv = self.qkv_proj(x)
        qkv = qkv.permute(0, 2, 4, 3, 1)#BDWHC
        qkv = qkv.reshape(B*D*W, H, self.heads, 3*self.headc)

        # print(f'galerkin after {qkv.shape}', flush=True)

        qkv = qkv.permute(0, 2, 1, 3)  #B*D*W, self.heads, H, 3*self.headc
        q, k, v = qkv.chunk(3, dim=-1) #W*B*D, heads, H, headc

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2,-1), v/(H)) #W*B*D, heads, headc, headc
        v = torch.matmul(q, v)  #B*D*W, heads, H, headc
        v = v.permute(0, 2, 1, 3).reshape(B, D, W, H, C) #B, D, W, H, C

        ret = v.permute(0, 4, 1, 3, 2) + bias #B, C, D, H, W
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias 
        
        return bias

class simple_attn_3d5(nn.Module):
    def __init__(self, midc, featc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc
        self.featc = featc
        self.kv_proj1 = nn.Conv2d(featc, midc, 1)
        self.kv_proj2 = nn.Conv3d(midc, 2 * midc, 1)

        self.q_proj = nn.Conv3d(midc, midc, 1)
        self.o_proj1 = nn.Conv3d(midc, midc, 1)
        self.o_proj2 = nn.Conv3d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = GELU()
    
    def forward(self, x, feat, name='0'):
        B, C, D, H, W = x.shape
        bias = x
        
        # print(f'galerkin before {x.shape}', flush=True)
        
        q = self.q_proj(x)
        feat = self.kv_proj1(feat)[:, :, D, :, :]
        feat = self.kv_proj2(feat)#B 2C D H W
        kv = feat.permute(0, 2, 3, 4, 1)
        kv = kv.reshape(B*D*H, W, self.heads, 2*self.headc)
        q  = q.reshape(B*D*H, W, self.heads, self.headc)
        # print(f'galerkin after {qkv.shape}', flush=True)
        q = q.permute(0, 2, 1, 3)  #B*D*H, self.heads, W, self.headc
        kv = kv.permute(0, 2, 1, 3)  #B*D*H, self.heads, W, 2*self.headc
        k, v = kv.chunk(2, dim=-1) #H*B*D, heads, W, headc

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2,-1), v/(W)) #H*B*D, heads, headc, headc
        v = torch.matmul(q, v)  #B*D*H, heads, W, headc
        v = v.permute(0, 2, 1, 3).reshape(B, D, H, W, C) #B, D, H, W, C

        ret = v.permute(0, 4, 1, 2, 3) + bias #B, C, D, H, W
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias 
        
        return bias

import torch
import torch.nn as nn
import torch.nn.functional as F


class simple_attn_3d_hw_patch(nn.Module):
    """
    Galerkin attention inside n*n patches on (H,W) for each disparity slice (D fixed).

    Input : x [B, C, D, H, W]
    Output: y [B, C, D, H, W]

    - Tokens are spatial positions inside each n*n patch (L = n*n).
    - For each (b, d, patch_id), do Galerkin:
        A = K^T (V / L)   -> [headc, headc]
        Y = Q A           -> [L, headc]
    - No softmax; linear/galerkin-style.
    """

    def __init__(self, midc: int, heads: int, patch: int = 4, stride: int = None, pad_mode: str = "replicate"):
        super().__init__()
        assert midc % heads == 0, "midc must be divisible by heads"
        self.midc = midc
        self.heads = heads
        self.headc = midc // heads

        self.patch = int(patch)
        self.stride = int(stride) if stride is not None else int(patch)
        self.pad_mode = pad_mode

        self.qkv_proj = nn.Conv3d(midc, 3 * midc, kernel_size=1, bias=True)
        self.o_proj1 = nn.Conv3d(midc, midc, kernel_size=1, bias=True)
        self.o_proj2 = nn.Conv3d(midc, midc, kernel_size=1, bias=True)

        # robust LN: normalize last dim (headc) per token per head
        self.kln = nn.LayerNorm(self.headc, elementwise_affine=True)
        self.vln = nn.LayerNorm(self.headc, elementwise_affine=True)

        self.act = nn.GELU()

    def _pad_hw(self, x: torch.Tensor):
        # Pad H,W so that unfold covers full map for given stride
        B, C, D, H, W = x.shape
        s = self.stride
        pad_h = (s - (H % s)) % s
        pad_w = (s - (W % s)) % s
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0, H, W)

        # F.pad for 5D: (Wl, Wr, Hl, Hr, Dl, Dr)
        x_pad = F.pad(x, (0, pad_w, 0, pad_h, 0, 0), mode=self.pad_mode)
        return x_pad, (pad_h, pad_w, H, W)

    def forward(self, x: torch.Tensor, name: str = "0"):
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,D,H,W], got {tuple(x.shape)}")

        B, C, D, H, W = x.shape
        bias = x

        # pad spatial dims
        x_pad, (pad_h, pad_w, H0, W0) = self._pad_hw(x)
        _, _, _, Hp, Wp = x_pad.shape

        # ---- QKV projection on 3D volume ----
        qkv = self.qkv_proj(x_pad)  # [B, 3C, D, Hp, Wp]

        # ---- Process HW patches per disparity slice ----
        # treat each disparity slice as separate 2D map:
        # [B, 3C, D, Hp, Wp] -> [B*D, 3C, Hp, Wp]
        qkv2d = qkv.permute(0, 2, 1, 3, 4).reshape(B * D, 3 * C, Hp, Wp)

        n = self.patch
        s = self.stride
        L = n * n

        # unfold: [BD, 3C*L, Np]
        patches = F.unfold(qkv2d, kernel_size=n, stride=s)  # [BD, 3C*L, Np]
        BD, threeC_L, Np = patches.shape
        if threeC_L != 3 * C * L:
            raise RuntimeError(f"Unexpected unfold shape: got {threeC_L}, expected {3*C*L}")

        # reshape to tokens per patch:
        # [BD, 3C, L, Np] -> [BD, Np, L, 3C]
        patches = patches.view(BD, 3 * C, L, Np).permute(0, 3, 2, 1).contiguous()

        # split heads and qkv:
        # [BD, Np, L, 3C] -> [BD*Np, heads, L, 3*headc]
        patches = patches.view(BD * Np, L, self.heads, 3 * self.headc).permute(0, 2, 1, 3)
        q, k, v = patches.chunk(3, dim=-1)  # each: [BD*Np, heads, L, headc]

        # LN on last dim
        k = self.kln(k)
        v = self.vln(v)

        # Galerkin (linear) attention within patch
        # A: [BD*Np, heads, headc, headc]
        A = torch.matmul(k.transpose(-2, -1), v / float(L))
        # Y: [BD*Np, heads, L, headc]
        y = torch.matmul(q, A)

        # merge heads back:
        # [BD*Np, heads, L, headc] -> [BD, Np, L, C]
        y = y.permute(0, 2, 1, 3).contiguous().view(BD * Np, L, C)
        y = y.view(BD, Np, L, C)

        # fold back to 2D:
        # [BD, Np, L, C] -> [BD, C*L, Np]
        y_fold_in = y.permute(0, 3, 2, 1).contiguous().view(BD, C * L, Np)
        y2d = F.fold(y_fold_in, output_size=(Hp, Wp), kernel_size=n, stride=s)  # [BD, C, Hp, Wp]

        # restore 3D volume: [BD, C, Hp, Wp] -> [B, C, D, Hp, Wp]
        y3d = y2d.view(B, D, C, Hp, Wp).permute(0, 2, 1, 3, 4).contiguous()

        # crop padding
        if pad_h > 0 or pad_w > 0:
            y3d = y3d[:, :, :, :H0, :W0]

        # residual + FFN + residual (same pattern as your original)
        ret = y3d + bias
        out = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        return out




class simple_attn_1d(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv1d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv1d(midc, midc, 1)
        self.o_proj2 = nn.Conv1d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = GELU()
    
    def forward(self, x, name='0'):
        B, C, W = x.shape
        bias = x

        qkv = self.qkv_proj(x).permute(0, 2, 1).reshape(B, W, self.heads, 3*self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2,-1), v) / (W)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, W, C)

        ret = v.permute(0, 2, 1) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias

class FlashMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, window_size=(-1,-1)):
        """
        @query: (B,L,C)
        """
        B,L,C = query.shape
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim)
        K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim)

        attn_output = F.scaled_dot_product_attention(Q, K, V)

        attn_output = attn_output.reshape(B,L,-1)
        output = self.out_proj(attn_output)

        return output
        
class FlashAttentionTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, act=nn.GELU, norm=nn.LayerNorm):
        super().__init__()
        self.self_attn = FlashMultiheadAttention(embed_dim, num_heads)
        self.act = act()

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, window_size=(-1, -1)):
        src2 = self.self_attn(src, src, src, src_mask, window_size=window_size)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class FlashAttentionTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, act=nn.GELU, norm=nn.LayerNorm):
        super().__init__()
        self.self_attn = FlashMultiheadAttention(embed_dim, num_heads)
        self.act = act()

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, window_size=(-1, -1)):
        src2 = self.self_attn(src, src, src, src_mask, window_size=window_size)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src  

class PositionalEmbedding(nn.Module):
  def __init__(self, d_model, max_len=512):
    super().__init__()

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model).float()
    pe.require_grad = False

    position = torch.arange(0, max_len).float().unsqueeze(1)  #(N,1)
    div_term = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()[None]

    pe[:, 0::2] = torch.sin(position * div_term)  #(N, d_model/2)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    self.pe = pe
    # self.register_buffer('pe', pe)  #(1, max_len, D)


  def forward(self, x, resize_embed=False):
    '''
    @x: (B,N,D)
    '''
    self.pe = self.pe.to(x.device).to(x.dtype)
    pe = self.pe
    if pe.shape[1]<x.shape[1]:
      if resize_embed:
        pe = F.interpolate(pe.permute(0,2,1), size=x.shape[1], mode='linear', align_corners=False).permute(0,2,1)
      else:
        raise RuntimeError(f'x:{x.shape}, pe:{pe.shape}')
    return x + pe[:, :x.size(1)]

class CostVolumeDisparityAttention(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, act=nn.GELU, norm_first=False, num_transformer=6, max_len=512, resize_embed=False):
    super().__init__()
    self.resize_embed = resize_embed
    self.sa = nn.ModuleList([])
    for _ in range(num_transformer):
      self.sa.append(FlashAttentionTransformerEncoderLayer(embed_dim=d_model, num_heads=nhead, dim_feedforward=dim_feedforward, act=act, dropout=dropout))
    self.pos_embed0 = PositionalEmbedding(d_model, max_len=max_len)


  def forward(self, cv, window_size=(-1,-1)):
    """
    @cv: (B,C,D,H,W) where D is max disparity
    """
    x = cv
    B,C,D,H,W = x.shape
    x = x.permute(0,3,4,2,1).reshape(B*H*W, D, C)
    x = self.pos_embed0(x, resize_embed=self.resize_embed)  #!NOTE No resize since disparity is pre-determined
    for i in range(len(self.sa)):
        x = self.sa[i](x, window_size=window_size)
    x = x.reshape(B,H,W,D,C).permute(0,4,3,1,2)

    return x