import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch import nn
from mamba_ssm.modules.mamba_new import Mamba as FABS6


class EmbeddingModule(nn.Module):
    def __init__(self, 
                 input_c: int, 
                 emb_c: int, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int, 
                 dilation: int, 
                 groups: int, 
                 bias: bool, 
                 padding_mode: str):
        super().__init__()
        assert (kernel_size%2 == 1) and (kernel_size//2 == padding) # element must be aligned
        self.stride = stride
        
        self.conv_1 = nn.Conv1d(
            in_channels=input_c, 
            out_channels=emb_c, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias,
            padding_mode=padding_mode)
        self.ln_1 = nn.LayerNorm(emb_c)
        self.act_1 = nn.ReLU(inplace=True)
        
        self.conv_2 = nn.Conv1d(
            in_channels=emb_c, 
            out_channels=emb_c, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias,
            padding_mode=padding_mode)
        self.ln_2 = nn.LayerNorm(emb_c)
        self.act_2 = nn.ReLU(inplace=True)
        
        # zero out the bias term if it exists
        if bias: 
            nn.init.constant_(self.conv_1.bias, 0.)
            nn.init.constant_(self.conv_2.bias, 0.)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple:
        # x: batch size, feature channel, sequence length
        # mask: batch size, 1, sequence length (bool)
        B,C,T = x.size()
        mask = mask.to(x.dtype)
        assert T%self.stride == 0 # input length must be divisible by stride
        
        x = self.conv_1(x)
        x = x * mask.detach()
        mask = mask.bool()
        
        x = rearrange(x, 'b c l -> b l c')
        x = self.ln_1(x)
        x = rearrange(x, 'b l c -> b c l')
        x = self.act_1(x)
        
        x = self.conv_2(x)
        x = x * mask.detach()
        mask = mask.bool()
        
        x = rearrange(x, 'b c l -> b l c')
        x = self.ln_2(x)
        x = rearrange(x, 'b l c -> b c l')
        x = self.act_2(x)
        
        return x, mask


class StemModule(nn.Module):
    def __init__(self, 
                 block_n: int, 
                 emb_c: int, 
                 kernel_size: int, 
                 drop_path_rate: float, 
                 recurrent: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.recurrent = recurrent
        
        for _ in range(block_n):
            ln = nn.LayerNorm(emb_c)
            t_mamba = FABS6(d_model=emb_c, 
                            d_conv=kernel_size, 
                            use_fast_path=True, 
                            expand=1, 
                            d_state=16, 
                            num_kernels=3)
            c_mamba = FABS6(d_model=2304//4, 
                            d_conv=kernel_size, 
                            use_fast_path=True, 
                            expand=1, 
                            d_state=16, 
                            num_kernels=3)
            drop_path = AffineDropPath(num_dim=emb_c, 
                                       drop_prob=drop_path_rate)
            adaptive_pool = nn.AdaptiveAvgPool1d(output_size=2304//4)
            final_pool = nn.AvgPool1d(kernel_size=2304//4)
            
            self.blocks.append(nn.ModuleDict({
                'ln': ln,
                't_mamba': t_mamba,
                'c_mamba': c_mamba,
                'drop_path': drop_path,
                'adaptive_pool': adaptive_pool,
                'final_pool': final_pool,
                'sigmoid': nn.Sigmoid()
            }))
        torch.use_deterministic_algorithms(True, warn_only=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple:
        for block in self.blocks:
            res = x
            for _ in range(self.recurrent):
                x_t = rearrange(x, 'b c l -> b l c')
                x_t = block['ln'](x_t)
                x_t = block['t_mamba'](x_t)
                x_t = rearrange(x_t, 'b l c -> b c l')
                x_t = x_t * mask.to(x.dtype)
                
                with torch.backends.cudnn.flags(enabled=False):
                    x_c = block['adaptive_pool'](x)  # b c l -> b c l//4
                x_c = block['c_mamba'](x_c)
                x_c = block['final_pool'](x_c)  # b c l//4 -> b c 1 using AvgPool1d
                x_c = block['sigmoid'](x_c)  # Apply sigmoid after c_mamba
                x_c = F.interpolate(x_c, size=x_t.size(-1), mode='nearest')  # b c l//4 -> b c l
                x_c = x_c * mask.to(x.dtype)
                
                x = x_t * x_c  # SE block mechanism
                x = block['drop_path'](x)
                x = x + res
        return x, mask


class BranchModule(nn.Module):
    def __init__(self,
                 block_n: int,
                 emb_c: int,
                 kernel_size: int,
                 drop_path_rate: float):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_n):
            ln = nn.LayerNorm(emb_c)
            mamba = FABS6(d_model=emb_c, 
                        d_conv=kernel_size, 
                        use_fast_path=True, 
                        expand=1, 
                        d_state=16)
            drop_path = AffineDropPath(num_dim=emb_c, drop_prob=drop_path_rate)
            ds_maxpool = MaxPooler(kernel_size=3, stride=2, padding=1)  # l -> 0.5l
            
            self.blocks.append(nn.ModuleDict({
                'mamba': mamba,
                'ln': ln,
                'drop_path': drop_path,
                'ds_maxpool': ds_maxpool
            }))
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple:
        out_feats, out_masks = tuple(), tuple()
        out_feats += (x, )
        out_masks += (mask,)
        
        for block in self.blocks:
            res = x
            x = rearrange(x, 'b c l -> b l c')
            x = block['ln'](x)
            x = block['mamba'](x)
            x = rearrange(x, 'b l c -> b c l')
            x = x * mask.to(x.dtype)
            x = block['drop_path'](x)
            x = x + res
            x, mask = block['ds_maxpool'](x, mask)
            out_feats += (x, )
            out_masks += (mask,)
        
        return out_feats, out_masks


# helper functions for Transformer blocks
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return rearrange(torch.FloatTensor(sinusoid_table), 't c -> 1 c t')



# drop path: from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
class Scale(nn.Module):
    """ Multiply the output regression range by a learnable constant value """
    def __init__(self, init_value=1.0):
        """ init_value : initial value for the scalar """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32),requires_grad=True)
    
    def forward(self, x):
        return x*self.scale


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """ Stochastic Depth per sample. """
    if drop_prob == 0.0 or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,)*(x.ndim-1)  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob)*mask
    return output


class DropPath(nn.Module):
    """ Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks). """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """ 
        Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init) See: https://arxiv.org/pdf/2103.17239.pdf """
    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(init_scale_value*torch.ones((1,num_dim,1)),requires_grad=True)
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(self.scale*x, self.drop_prob, self.training)


class MaxPooler(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.ds_pooling = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        self.stride = stride
    
    def forward(self, x, mask, **kwargs):
        out_mask = F.interpolate(mask.to(x.dtype), size=x.size(-1)//self.stride, mode='nearest') if self.stride > 1 else mask
        out = self.ds_pooling(x)*out_mask.to(x.dtype)
        return out, out_mask.bool()


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(1, dim, 1))

    def forward(self, x):
        return x * self.scale


class MaskedConv1D(nn.Module):
    """ Masked 1D convolution. Interface remains the same as Conv1d.
        Only support a subset of 1d convs """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 groups=1, 
                 bias=True, 
                 padding_mode='zeros'):
        super().__init__()
        assert (kernel_size%2 == 1) and (kernel_size//2 == padding) # element must be aligned
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias: nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B,C,T = x.size()
        # input length must be divisible by stride
        assert T%self.stride == 0
        
        # conv
        out_conv = self.conv(x)
        out_mask = F.interpolate(mask.to(x.dtype), size=T//self.stride, mode='nearest') if self.stride > 1 else mask.to(x.dtype)
        
        # masking the output, stop grad to mask
        out_conv = out_conv*out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """ LayerNorm that supports inputs of size B, C, T """
    def __init__(self, num_channels, eps=1e-5, affine=True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels
        
        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma+self.eps)
        
        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias
            
        return out