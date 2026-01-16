from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from einops import rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, DropPath
import numpy as np
import math

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

import os 
def debug_save(tensor, name, mode):
    # mode is 'enc' or 'dec'
    # Save to a folder so we don't clutter root
    os.makedirs(f"./debug_{mode}", exist_ok=True)
    # Save as .pt
    torch.save(tensor.detach().cpu(), f"./debug_{mode}/{name}.pt")

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels)) 

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x 

def update_registered_buffers(module, module_name, buffer_names, state_dict, policy="resize_if_empty", dtype=torch.int):
    if not module: return
    for buffer_name in buffer_names:
        key = f"{module_name}.{buffer_name}"
        if key in state_dict:
            new_size = state_dict[key].size()
            buf = getattr(module, buffer_name)
            if buf is not None and buf.size() != new_size:
                buf.resize_(new_size)

# --- Blocks ---
class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = min(in_ch, out_ch) // 2
        self.conv1 = conv1x1(in_ch, mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_ch, mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(mid_ch, out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)
        out = x
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return out + identity

class ResidualBottleneckBlockWithStride(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv =  conv(in_ch, out_ch, kernel_size=5, stride=2)
        self.res1 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res2 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res3 = ResidualBottleneckBlock(out_ch, out_ch)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        return out

class ResidualBottleneckBlockWithUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.res1 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res2 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res3 = ResidualBottleneckBlock(in_ch, in_ch)
        self.conv = deconv(in_ch, out_ch, kernel_size=5, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)
        out = self.conv(out)
        return out

class WMSA(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5 
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True) 
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask
        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2)) 
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))
        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)
        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b h w c')
        return x

class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = hidden_features//2
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x)) * v
        x = self.fc2(x)
        return x

class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale

class ResScaleConvolutionGateBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = ConvolutionalGLU(input_dim, input_dim * 4)
        self.res_scale_1 = Scale(input_dim, init_value=1.0)
        self.res_scale_2 = Scale(input_dim, init_value=1.0)

    def forward(self, x):
        x = self.res_scale_1(x) + self.drop_path(self.msa(self.ln1(x)))
        x = self.res_scale_2(x) + self.drop_path(self.mlp(self.ln2(x)))
        return x

class SwinBlockWithConvMulti(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, block=ResScaleConvolutionGateBlock, block_num=2, **kwargs) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.block_num = block_num
        for i in range(block_num):
            ty = 'W' if i%2==0 else 'SW'
            self.layers.append(block(input_dim, input_dim, head_dim, window_size, drop_path, type=ty))
        self.conv = conv(input_dim, output_dim, 3, 1)
        self.window_size = window_size

    def forward(self, x):
        if (x.size(-1) <= self.window_size) or (x.size(-2) <= self.window_size):
            padding_row = (self.window_size - x.size(-2)) // 2
            padding_col = (self.window_size - x.size(-1)) // 2
            x = F.pad(x, (padding_col, padding_col+1, padding_row, padding_row+1))
        trans_x = Rearrange('b c h w -> b h w c')(x)
        for i in range(self.block_num):
            trans_x = self.layers[i](trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        trans_x = self.conv(trans_x)
        return trans_x + x

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ConvWithDW(nn.Module):
    def __init__(self, input_dim=320, output_dim=320):
        super(ConvWithDW, self).__init__()
        self.in_trans = nn.Conv2d(input_dim, output_dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.act1 = nn.GELU()
        self.dw_conv = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, stride=1, groups=output_dim, bias=True)
        self.act2 = nn.GELU()
        self.out_trans = nn.Conv2d(output_dim, output_dim, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        x = self.in_trans(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        x = self.act2(x)
        x = self.out_trans(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, dim=320):
        super(DenseBlock, self).__init__()
        self.layer_num = 3
        self.conv_layers = nn.ModuleList([
            nn.Sequential(nn.GELU(), ConvWithDW(dim, dim)) for i in range(self.layer_num)  
        ])
        self.proj = nn.Conv2d(dim*(self.layer_num+1), dim, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        outputs = [x]
        for i in range(self.layer_num):
            outputs.append(self.conv_layers[i](outputs[-1]))
        x = self.proj(torch.cat(outputs, dim=1))
        return x

class MultiScaleAggregation(nn.Module):
    def __init__(self, dim):
        super(MultiScaleAggregation, self).__init__()
        self.s = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.spatial_atte = SpatialAttentionModule()
        self.dense = DenseBlock(dim)
        
    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        s = self.s(x)
        s_out = self.dense(s)
        x = s_out * self.spatial_atte(s_out)
        x = rearrange(x, 'b c h w -> b h w c')
        return x

class MutiScaleDictionaryCrossAttentionGLU(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_rate=4, head_num=20, qkv_bias=True):
        super().__init__()
        dict_dim = 32 * head_num    
        self.head_num = head_num    
        self.scale = nn.Parameter(torch.ones(head_num, 1, 1))
        self.x_trans = nn.Linear(input_dim, dict_dim, bias=qkv_bias)
        self.ln_scale = nn.LayerNorm(dict_dim)
        self.msa = MultiScaleAggregation(dict_dim)
        self.lnx = nn.LayerNorm(dict_dim)
        self.q_trans = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.dict_ln = nn.LayerNorm(dict_dim)
        self.k = nn.Linear(dict_dim,dict_dim, bias=qkv_bias)
        self.linear = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.ln_mlp = nn.LayerNorm(dict_dim)
        self.mlp = ConvolutionalGLU(dict_dim, mlp_rate * dict_dim)
        self.output_trans = nn.Sequential(nn.Linear(dict_dim, output_dim))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.res_scale_1 = Scale(dict_dim, init_value=1.0)
        self.res_scale_2 = Scale(dict_dim, init_value=1.0)
        self.res_scale_3 = Scale(dict_dim, init_value=1.0)

    def forward(self, x, dt):
        B, C, H, W = x.size() 
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.x_trans(x)
        x = self.msa(self.ln_scale(x)) + self.res_scale_1(x)
        shortcut = x
        x = self.lnx(x)
        x = self.q_trans(x)
        x = rearrange(x, 'b h w c -> b c h w')
        q = rearrange(x, 'b (e c) h w -> b e (h w) c', e=self.head_num)
        dt = dt.to(x.device)
        dt = self.dict_ln(dt)
        k = self.k(dt)
        k = rearrange(k, 'b n (e c) -> b e n c', e=self.head_num)
        dt = rearrange(dt, 'b n (e c) -> b e n c', e=self.head_num)
        self.scale = self.scale.to(q.device)
        sim = torch.einsum('benc,bedc->bend', q, k)
        sim = sim * self.scale
        probs = self.softmax(sim)
        output = torch.einsum('bend,bedc->benc', probs, dt)
        output = rearrange(output, 'b e (h w) c -> b h w (e c) ', h = H, w = W)
        output = self.linear(output) + self.res_scale_2(shortcut)
        output = self.mlp(self.ln_mlp(output)) + self.res_scale_3(output)
        output = self.output_trans(output)
        output = rearrange(output, 'b h w c -> b c h w', )
        return output

# --- Models ---

class CompressModel(CompressionModel):
    """Standalone compression model (Master/Encoder)"""
    def __init__(self, head_dim=[8, 16, 32], drop_path_rate=0, N=192, M=320, num_slices=5, max_support_slices=5, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        self.N = N
        self.M = M
        
        input_image_channel = 3
        feature_dim = [96, 144, 256]
        basic_block = [ResScaleConvolutionGateBlock, ResScaleConvolutionGateBlock, ResScaleConvolutionGateBlock]
        swin_block = [SwinBlockWithConvMulti, SwinBlockWithConvMulti, SwinBlockWithConvMulti]
        block_num = [1, 2, 12]

        dict_num = 128
        dict_head_num = 20 
        dict_dim = 32 * dict_head_num
        self.dt = nn.Parameter(torch.randn([dict_num, dict_dim]), requires_grad=True)
        
        prior_dim = M
        mlp_rate = 4
        qkv_bias = True
        self.dt_cross_attention = nn.ModuleList([
            MutiScaleDictionaryCrossAttentionGLU(
                input_dim=M*2+(M//self.num_slices)*i, 
                output_dim=M, 
                head_num=dict_head_num, 
                mlp_rate=mlp_rate, 
                qkv_bias=qkv_bias
            ) for i in range(num_slices)
        ])

        self.m_down1 = [swin_block[0](feature_dim[0], feature_dim[0], self.head_dim[0], self.window_size, 0, basic_block[0], block_num=block_num[0])] + \
                      [ResidualBottleneckBlockWithStride(feature_dim[0], feature_dim[1])]
        self.m_down2 = [swin_block[1](feature_dim[1], feature_dim[1], self.head_dim[1], self.window_size, 0, basic_block[1], block_num=block_num[1])] + \
                      [ResidualBottleneckBlockWithStride(feature_dim[1], feature_dim[2])]
        self.m_down3 = [swin_block[2](feature_dim[2], feature_dim[2], self.head_dim[2], self.window_size, 0, basic_block[2], block_num=block_num[2])] + \
                      [conv(feature_dim[2], M, kernel_size=5, stride=2)] 
        
        self.g_a = nn.Sequential(
            *[ResidualBottleneckBlockWithStride(input_image_channel, feature_dim[0])] + 
            self.m_down1 + self.m_down2 + self.m_down3
        )

        self.ha_down = [SwinBlockWithConvMulti(N, N, 32, 4, 0, ResScaleConvolutionGateBlock, block_num=1)] + \
                      [conv(N, 192, kernel_size=3, stride=2)] 
        self.h_a = nn.Sequential(
            *[ResidualBottleneckBlockWithStride(M, N)] + 
            self.ha_down 
        )

        self.hs_up1 = [SwinBlockWithConvMulti(N, N, 32, 4, 0, ResScaleConvolutionGateBlock, block_num=1)] + \
                      [ResidualBottleneckBlockWithUpsample(N, M)]
        self.h_z_s1 = nn.Sequential(
            *[deconv(192, N, kernel_size=3, stride=2)] + 
            self.hs_up1
        )
        
        self.hs_up2 = [SwinBlockWithConvMulti(N, N, 32, 4, 0, ResScaleConvolutionGateBlock, block_num=1)] + \
                      [ResidualBottleneckBlockWithUpsample(N, M)]
        self.h_z_s2 = nn.Sequential(
            *[deconv(192, N, kernel_size=3, stride=2)] + 
            self.hs_up2
        )

        self.cc_mean_transforms = nn.ModuleList([
            nn.Sequential(
                conv(320*2 + (320//self.num_slices)*min(i, 5) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices) 
        ])
        
        self.cc_scale_transforms = nn.ModuleList([
            nn.Sequential(
                conv(320*2 + (320//self.num_slices)*min(i, 5) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        ])

        self.lrp_transforms = nn.ModuleList([
            nn.Sequential(
                conv(320*2 + (320//self.num_slices)*min(i+1, 6) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        ])

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        self.entropy_bottleneck.update(force=force)
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def forward(self, x):
        """Training Forward Pass"""
        b = x.size(0)
        dt = self.dt.repeat([b, 1, 1])
        dt = dt.to(x.device)
        y = self.g_a(x) 
        y_shape = y.shape[2:]
        
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        
        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        scale_list = []
        
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            query = torch.cat([latent_scales] + [latent_means] + support_slices, dim=1)
            dict_info = self.dt_cross_attention[slice_index](query, dt)
            support = torch.cat([query] + [dict_info], dim=1)
            
            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_list.append(mu)
            
            scale = self.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            scale_list.append(scale)
            
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        scales = torch.cat(scale_list, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        
        return {
            "y_hat": y_hat,
            "z_hat": z_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para": {"means": means, "scales": scales, "y": y}
        }

    def compress(self, x):
        """Inference Compression"""
        b = x.size(0)
        dt = self.dt.repeat([b, 1, 1]).to(x.device)
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        debug_save(z_hat, "00_z_hat", "enc") # Check root stability

        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        debug_save(latent_scales, "01_latent_scales", "enc") # Check NN processing
        debug_save(latent_means,  "01_latent_means",  "enc")

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            query = torch.cat([latent_scales] + [latent_means] + support_slices, dim=1)
            dict_info = self.dt_cross_attention[slice_index](query, dt)
            debug_save(dict_info, f"02_dict_info_slice{slice_index}", "enc") # Check Attention stability

            support = torch.cat([query] + [dict_info], dim=1)
            
            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            
            scale = self.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            debug_save(mu,    f"03_mu_slice{slice_index}",    "enc") # Check drift
            debug_save(scale, f"03_scale_slice{slice_index}", "enc") # Check crash cause

            if slice_index == 0:
                torch.save(scale.cpu(), f"debug_scale_enc_slice0.pt")

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu
            debug_save(y_hat_slice, f"04_y_hat_slice{slice_index}", "enc") # Check accumulation

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)


class DecompressModel(CompressionModel):
    """Standalone decompression model (Worker/Decoder)"""
    def __init__(self, head_dim=[32, 16, 8], drop_path_rate=0, N=192, M=320, num_slices=5, max_support_slices=5, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        self.N = N
        self.M = M
        
        output_image_channel = 3
        feature_dim = [96, 144, 256]
        basic_block = [ResScaleConvolutionGateBlock, ResScaleConvolutionGateBlock, ResScaleConvolutionGateBlock]
        swin_block = [SwinBlockWithConvMulti, SwinBlockWithConvMulti, SwinBlockWithConvMulti]
        block_num = [1, 2, 12]

        # Replicated Shared Components
        dict_num = 128
        dict_head_num = 20 
        dict_dim = 32 * dict_head_num
        self.dt = nn.Parameter(torch.randn([dict_num, dict_dim]), requires_grad=True)
        
        prior_dim = M
        mlp_rate = 4
        qkv_bias = True
        self.dt_cross_attention = nn.ModuleList([
            MutiScaleDictionaryCrossAttentionGLU(
                input_dim=M*2+(M//self.num_slices)*i, 
                output_dim=M, 
                head_num=dict_head_num, 
                mlp_rate=mlp_rate, 
                qkv_bias=qkv_bias
            ) for i in range(num_slices)
        ])

        self.m_up1 = [swin_block[2](feature_dim[2], feature_dim[2], self.head_dim[0], self.window_size, 0, basic_block[2], block_num=block_num[2])] + \
                      [ResidualBottleneckBlockWithUpsample(feature_dim[2], feature_dim[1])]
        self.m_up2 = [swin_block[1](feature_dim[1], feature_dim[1], self.head_dim[1], self.window_size, 0, basic_block[1], block_num=block_num[1])] + \
                      [ResidualBottleneckBlockWithUpsample(feature_dim[1], feature_dim[0])]
        self.m_up3 = [swin_block[0](feature_dim[0], feature_dim[0], self.head_dim[2], self.window_size, 0, basic_block[0], block_num=block_num[0])] + \
                      [ResidualBottleneckBlockWithUpsample(feature_dim[0], output_image_channel)]
        
        self.g_s = nn.Sequential(
            *[deconv(M, feature_dim[2], kernel_size=5, stride=2)] + 
            self.m_up1 + self.m_up2 + self.m_up3
        )

        self.hs_up1 = [SwinBlockWithConvMulti(N, N, 32, 4, 0, ResScaleConvolutionGateBlock, block_num=1)] + \
                      [ResidualBottleneckBlockWithUpsample(N, M)]
        self.h_z_s1 = nn.Sequential(
            *[deconv(192, N, kernel_size=3, stride=2)] + 
            self.hs_up1
        )
        
        self.hs_up2 = [SwinBlockWithConvMulti(N, N, 32, 4, 0, ResScaleConvolutionGateBlock, block_num=1)] + \
                      [ResidualBottleneckBlockWithUpsample(N, M)]
        self.h_z_s2 = nn.Sequential(
            *[deconv(192, N, kernel_size=3, stride=2)] + 
            self.hs_up2
        )

        self.cc_mean_transforms = nn.ModuleList([
            nn.Sequential(
                conv(320*2 + (320//self.num_slices)*min(i, 5) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices) 
        ])
        
        self.cc_scale_transforms = nn.ModuleList([
            nn.Sequential(
                conv(320*2 + (320//self.num_slices)*min(i, 5) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        ])

        self.lrp_transforms = nn.ModuleList([
            nn.Sequential(
                conv(320*2 + (320//self.num_slices)*min(i+1, 6) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        ])

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        self.entropy_bottleneck.update(force=force)
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def forward(self, y_hat, z_hat):
        """Training Forward Pass"""
        b = z_hat.size(0)
        # Ensure dt is on the correct device (GPU)
        dt = self.dt.repeat([b, 1, 1]).to(z_hat.device)
        
        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        y_shape = y_hat.shape[2:]

        y_slices = y_hat.chunk(self.num_slices, 1)
        y_hat_slices = []
        
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            query = torch.cat([latent_scales] + [latent_means] + support_slices, dim=1)
            dict_info = self.dt_cross_attention[slice_index](query, dt)
            support = torch.cat([query] + [dict_info], dim=1)
            
            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            
            lrp_support = torch.cat([support, y_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_slice + lrp
            y_hat_slices.append(y_hat_slice)

        y_hat_final = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat_final)
        
        return {"x_hat": x_hat}
    
    def decompress(self, strings, shape):
        """Inference Decompression"""
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        debug_save(z_hat, "00_z_hat", "dec") # Check root stability
        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        debug_save(latent_scales, "01_latent_scales", "dec") # Check NN processing
        debug_save(latent_means,  "01_latent_means",  "dec")
        
        b = z_hat.size(0)
        # Ensure dt is on same device as dynamic activations
        dt = self.dt.repeat([b, 1, 1]).to(z_hat.device)
        
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]
        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            query = torch.cat([latent_scales] + [latent_means] + support_slices, dim=1)
            dict_info = self.dt_cross_attention[slice_index](query, dt)
            debug_save(dict_info, f"02_dict_info_slice{slice_index}", "dec") # Check Attention stability
            support = torch.cat([query] + [dict_info], dim=1)
            
            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            
            scale = self.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            debug_save(mu,    f"03_mu_slice{slice_index}",    "dec") # Check drift
            debug_save(scale, f"03_scale_slice{slice_index}", "dec") # Check crash cause

            if slice_index == 0:
                torch.save(scale.cpu(), f"debug_scale_dec_slice0.pt")
            index = self.gaussian_conditional.build_indexes(scale)
            
            # Decode using CPU list, convert back to tensor on correct device
            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1]).to(z_hat.device)
            
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            debug_save(y_hat_slice, f"04_y_hat_slice{slice_index}", "dec") # Check accumulation
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

class ParameterSync:
    """Utility class for synchronizing parameters between CompressModel (Master) and DecompressModel (Replica)"""
    
    @staticmethod
    def sync_compress_to_decompress(compress_model, decompress_model):
        """Sync parameters from compression model to decompression model"""
        # 1. Sync the Dictionary (nn.Parameter)
        # Using copy_ with explicit data ensures PyTorch handles device transfer
        decompress_model.dt.data.copy_(compress_model.dt.data)
        
        # 2. Sync Shared Modules (Encoder parameters copied to Decoder replica)
        # load_state_dict handles device transfer automatically
        shared_modules = [
            ('dt_cross_attention', compress_model.dt_cross_attention),
            ('cc_mean_transforms', compress_model.cc_mean_transforms),
            ('cc_scale_transforms', compress_model.cc_scale_transforms),
            ('lrp_transforms', compress_model.lrp_transforms),
            ('h_z_s1', compress_model.h_z_s1),
            ('h_z_s2', compress_model.h_z_s2),
            ('entropy_bottleneck', compress_model.entropy_bottleneck),
            ('gaussian_conditional', compress_model.gaussian_conditional)
        ]
        
        for name, module in shared_modules:
            getattr(decompress_model, name).load_state_dict(module.state_dict())

    @staticmethod
    def save_shared_parameters(model, path):
        """Save only shared parameters"""
        shared_params = {
            'dt': model.dt.cpu(),
            'dt_cross_attention': [m.state_dict() for m in model.dt_cross_attention],
            'cc_mean_transforms': [m.state_dict() for m in model.cc_mean_transforms],
            'cc_scale_transforms': [m.state_dict() for m in model.cc_scale_transforms],
            'lrp_transforms': [m.state_dict() for m in model.lrp_transforms],
            'h_z_s1': model.h_z_s1.state_dict(),
            'h_z_s2': model.h_z_s2.state_dict(),
            'entropy_bottleneck': model.entropy_bottleneck.state_dict(),
            'gaussian_conditional': model.gaussian_conditional.state_dict(),
        }
        torch.save(shared_params, path)