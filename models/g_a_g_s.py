from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
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

from einops import rearrange ,repeat
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, DropPath
import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
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

class ResidualBottleneckBlock(nn.Module):
    """Residual bottleneck block.

    Introduced by [He2016], this block sandwiches a 3x3 convolution
    between two 1x1 convolutions which reduce and then restore the
    number of channels. This reduces the number of parameters required.

    [He2016]: `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/abs/1512.03385>`_, by Kaiming He, Xiangyu Zhang,
    Shaoqing Ren, and Jian Sun, CVPR 2016.

    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels
    """

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
    """ Self-attention module in Swin Transformer
    """

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
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
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
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
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
        resize = False
        if (x.size(-1) <= self.window_size) or (x.size(-2) <= self.window_size):
            padding_row = (self.window_size - x.size(-2)) // 2
            padding_col = (self.window_size - x.size(-1)) // 2
            x = F.pad(x, (padding_col, padding_col+1, padding_row, padding_row+1))
        trans_x = Rearrange('b c h w -> b h w c')(x)
        for i in range(self.block_num):
            trans_x = self.layers[i](trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        trans_x = self.conv(trans_x)
        if resize:
            x = F.pad(x, (-padding_col, -padding_col-1, -padding_row, -padding_row-1))
        return trans_x + x

from collections import OrderedDict

class SimpleAutoencoder(nn.Module):
    def __init__(self, head_dim=[8, 16, 32, 32, 16, 8], N=192, M=320, **kwargs):
        super(SimpleAutoencoder, self).__init__()
        
        # Model parameters
        self.head_dim = head_dim
        self.window_size = 8
        self.M = M
        
        # Feature dimensions and block configurations
        input_image_channel = 3
        output_image_channel = 3
        feature_dim = [96, 144, 256]
        
        basic_block = [ResScaleConvolutionGateBlock, ResScaleConvolutionGateBlock, ResScaleConvolutionGateBlock]
        swin_block = [SwinBlockWithConvMulti, SwinBlockWithConvMulti, SwinBlockWithConvMulti]
        block_num = [1, 2, 12]
        
        # Build encoder (g_a) - same as DCAE
        self.m_down1 = [swin_block[0](feature_dim[0], feature_dim[0], self.head_dim[0], self.window_size, 0, basic_block[0], block_num=block_num[0])] + \
                      [ResidualBottleneckBlockWithStride(feature_dim[0], feature_dim[1])]
        self.m_down2 = [swin_block[1](feature_dim[1], feature_dim[1], self.head_dim[1], self.window_size, 0, basic_block[1], block_num=block_num[1])] + \
                      [ResidualBottleneckBlockWithStride(feature_dim[1], feature_dim[2])]
        self.m_down3 = [swin_block[2](feature_dim[2], feature_dim[2], self.head_dim[2], self.window_size, 0, basic_block[2], block_num=block_num[2])] + \
                      [conv(feature_dim[2], M, kernel_size=5, stride=2)]
        
        self.g_a = nn.Sequential(*[ResidualBottleneckBlockWithStride(input_image_channel, feature_dim[0])] + 
                                self.m_down1 + self.m_down2 + self.m_down3)
        
        # Build decoder (g_s) - same as DCAE
        self.m_up1 = [swin_block[2](feature_dim[2], feature_dim[2], self.head_dim[3], self.window_size, 0, basic_block[2], block_num=block_num[2])] + \
                      [ResidualBottleneckBlockWithUpsample(feature_dim[2], feature_dim[1])]
        self.m_up2 = [swin_block[1](feature_dim[1], feature_dim[1], self.head_dim[4], self.window_size, 0, basic_block[1], block_num=block_num[1])] + \
                      [ResidualBottleneckBlockWithUpsample(feature_dim[1], feature_dim[0])]
        self.m_up3 = [swin_block[0](feature_dim[0], feature_dim[0], self.head_dim[5], self.window_size, 0, basic_block[0], block_num=block_num[0])] + \
                      [ResidualBottleneckBlockWithUpsample(feature_dim[0], output_image_channel)]
        
        self.g_s = nn.Sequential(*[deconv(M, feature_dim[2], kernel_size=5, stride=2)] + 
                                self.m_up1 + self.m_up2 + self.m_up3)
    
    def forward(self, x):
        """Forward pass: encode then decode"""
        latent = self.g_a(x)
        reconstructed = self.g_s(latent)
        return {
            "x_hat": reconstructed,
            "latent": latent
        }
    
    def compress(self, x):
        """Compress input image to latent representation"""
        self.eval()
        with torch.no_grad():
            latent = self.g_a(x)
        return latent
    
    def decompress(self, latent):
        """Decompress latent representation to reconstructed image"""
        self.eval()
        with torch.no_grad():
            reconstructed = self.g_s(latent)
            reconstructed = torch.clamp(reconstructed, 0, 1)  # Ensure valid pixel values
        return reconstructed
    
    def encode(self, x):
        """Alias for compress - for compatibility"""
        return self.compress(x)
    
    def decode(self, latent):
        """Alias for decompress - for compatibility"""
        return self.decompress(latent)

    def load_from_dcae(self, dcae_model_or_path, strict=False, ignore_shape_mismatch=True):
        """Load pretrained weights from DCAE model with shape mismatch handling"""
        if isinstance(dcae_model_or_path, str):
            checkpoint = torch.load(dcae_model_or_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                dcae_state_dict = checkpoint['state_dict']
            else:
                dcae_state_dict = checkpoint
        elif hasattr(dcae_model_or_path, 'state_dict'):
            dcae_state_dict = dcae_model_or_path.state_dict()
        else:
            dcae_state_dict = dcae_model_or_path
        
        simple_state_dict = OrderedDict()
        model_keys = set(self.state_dict().keys())
        current_state = self.state_dict()
        
        loaded_count = 0
        skipped_count = 0
        shape_mismatches = []
        
        for key, value in dcae_state_dict.items():
            # Remove 'module.' prefix if present
            clean_key = key.replace('module.', '', 1) if key.startswith('module.') else key
            
            # Check if this key matches our g_a or g_s structure
            if clean_key.startswith('g_a.') or clean_key.startswith('g_s.'):
                if clean_key in model_keys:
                    # Check for shape mismatch
                    if clean_key in current_state:
                        if value.shape != current_state[clean_key].shape:
                            shape_mismatches.append((clean_key, value.shape, current_state[clean_key].shape))
                            if ignore_shape_mismatch:
                                print(f"Skipping layer due to shape mismatch: {clean_key} "
                                    f"(source: {value.shape}, target: {current_state[clean_key].shape})")
                                skipped_count += 1
                                continue
                            else:
                                raise RuntimeError(f"Shape mismatch for {clean_key}: {value.shape} vs {current_state[clean_key].shape}")
                    
                    simple_state_dict[clean_key] = value
                    loaded_count += 1
        
        # Load the weights
        missing_keys, unexpected_keys = self.load_state_dict(simple_state_dict, strict=False)
        
        print(f"Successfully loaded {loaded_count} weights from DCAE.")
        print(f"Skipped {skipped_count} weights due to shape mismatches.")
        if shape_mismatches:
            print("Shape mismatches found:")
            for key, src_shape, tgt_shape in shape_mismatches:
                print(f"  {key}: {src_shape} -> {tgt_shape}")
        
        return self
    
    @classmethod
    def from_dcae(cls, dcae_model_or_path, ignore_shape_mismatch, **kwargs):
        """
        Create SimpleAutoencoder from pretrained DCAE model
        
        Args:
            dcae_model_or_path: Either a DCAE model instance or path to saved state dict
            **kwargs: Additional arguments for SimpleAutoencoder initialization
        """
        # Create new SimpleAutoencoder instance
        model = cls(**kwargs)
        
        # Load pretrained weights
        model.load_from_dcae(dcae_model_or_path, strict=False, ignore_shape_mismatch=ignore_shape_mismatch)
        
        return model
    
    def get_compression_ratio(self, x):
        """
        Calculate approximate compression ratio
        
        Args:
            x: Input image tensor
            
        Returns:
            Compression ratio (original_size / compressed_size)
        """
        if x.dtype == torch.float16:
            x_bit = 16
        elif x.dtype == torch.float32:
            x_bit = 32
        else:
            raise ValueError("Unsupported data type for compression ratio calculation")
        original_size = x.numel() * x_bit # Assuming 32-bit floats
        latent = self.compress(x)
        compressed_size = latent.numel() * 32  # Assuming 32-bit floats
        
        return original_size / compressed_size


## play with M

