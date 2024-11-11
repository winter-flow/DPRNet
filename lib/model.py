import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import sys
from mmcv.cnn import build_norm_layer
sys.path.insert(0, '../../')
from lib.pvtv2 import pvt_v2_b2

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def build_act_layer(act_type):
    """Build activation layer."""
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.GELU()
    else:
        return nn.GELU()

class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3,],
                 channel_split=[1, 3, 4,],
                ):
        super(MultiOrderDWConv, self).__init__()
        # dw_dilation=[1, 1, 2, 3], channel_split=[1, 1, 2, 4]

        self.split_ratio = [i / sum(channel_split) for i in channel_split]  # 分数, [1/8,1/8,2/8,4/8]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims) #分数 * 输入维度
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims) #分数 * 输入维度
        self.embed_dims_3 = int(self.split_ratio[3] * embed_dims) #分数 * 输入维度
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2 - self.embed_dims_3
        self.embed_dims = embed_dims
        # assert len(dw_dilation) == len(channel_split) == 3
        # assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )

        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        self.proj_1 = nn.Conv2d(self.embed_dims_1,2*self.embed_dims_1,kernel_size=3,stride=1,padding=1)
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        self.proj_2 = nn.Conv2d(self.embed_dims_2,2*self.embed_dims_2,kernel_size=3,stride=1,padding=1)
        # DW conv 3
        self.DW_conv3 = nn.Conv2d(
            in_channels=self.embed_dims_3,
            out_channels=self.embed_dims_3,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[3]) // 2,
            groups=self.embed_dims_3,
            stride=1, dilation=dw_dilation[3],
        )
        self.proj_3 = nn.Conv2d(self.embed_dims_3,2*self.embed_dims_3,kernel_size=3,stride=1,padding=1)
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)
        self.relu = nn.GELU()

    def forward(self, x):
        x = self.DW_conv0(x)

        x_0 = x[:, : self.embed_dims_0, ...] # 1/8
        x_1 = x[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...] # 1/8
        x_2 = x[:, self.embed_dims_0+self.embed_dims_1:self.embed_dims_0+self.embed_dims_1+self.embed_dims_2, ...] # 2/8
        x_3 = x[:, self.embed_dims-self.embed_dims_3:, ...] # 4/8
        
        f_x01 = x_0 + x_1
        f_x01 = self.DW_conv1(f_x01)
        f_x01 = self.relu(f_x01)
        f_x01 = self.proj_1(f_x01)

        f_x012 = f_x01 + x_2
        f_x012 = self.DW_conv2(f_x012)
        f_x012 = self.relu(f_x012)
        f_x012 = self.proj_2(f_x012)

        f_x0123 = f_x012 + x_3
        f_x0123 = self.DW_conv3(f_x0123)
        f_x0123 = self.relu(f_x0123)
        f_x0123 = self.proj_3(f_x0123)

        x = self.PW_conv(f_x0123)
        x = self.relu(x)

        return x

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,dilation=1):
        super(CBR, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,dilation=dilation)
        self.norm_cfg = {'type': 'BN', 'requires_grad': True}
        _, self.bn = build_norm_layer(self.norm_cfg, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)

        return x

class FeatureAggregation(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation.

    Args:
        embed_dims (int): Number of input channels.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for Spatial Block.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 embed_dims_2,
                 targed_dims,
                 attn_dw_dilation=[1, 1, 2, 3],
                 attn_channel_split=[1, 1, 2, 4],
                #  attn_act_type='SiLU',
                #  attn_act_type='ReLU',
                 attn_act_type='GELU',
                 attn_force_fp32=False,
                ):
        super(FeatureAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=targed_dims, out_channels=targed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=targed_dims, out_channels=targed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=targed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(
            in_channels=targed_dims, out_channels=targed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = build_act_layer(attn_act_type)
        self.act_gate = build_act_layer(attn_act_type)

        # decompose
        self.sigma = ElementScale(
            targed_dims, init_value=1e-5, requires_grad=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_1 = nn.Sequential(
            nn.Linear(targed_dims, targed_dims//2),
            nn.GELU(),
            nn.Linear(targed_dims//2, targed_dims),
            nn.Sigmoid()
        )
        self.channel_attention_2 = nn.Sequential(
            nn.Linear(targed_dims, targed_dims//2),
            nn.GELU(),
            nn.Linear(targed_dims//2, targed_dims),
            nn.Sigmoid()
        )

        self.feature_context_1 = nn.Sequential(
            nn.Conv2d(targed_dims, targed_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(targed_dims),
            nn.GELU()
        )

        self.feature_context_2 = nn.Sequential(
            nn.Conv2d(targed_dims, targed_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(targed_dims),
            nn.GELU()
        )
        self.cbr_1 = CBR(embed_dims, targed_dims, kernel_size=3, stride=1, dilation=1, padding=1)
        self.cbr_2 = CBR(embed_dims_2, targed_dims, kernel_size=3, stride=1, dilation=1, padding=1)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))

    # def forward(self, x):
    def forward(self, x, x_small):
        x = self.cbr_1(x)
        x_small = self.cbr_2(x_small)
        x_small = F.interpolate(x_small, size=x.size()[2:],  mode='bicubic',align_corners = False)
        B, C, _, _ = x.size()

        vec_y_1 = self.avg_pool(x).view(B, C)
        channel_att_1 = self.channel_attention_1(vec_y_1).view(B, C, 1, 1)
        x = x * channel_att_1 

        vec_y_2 = self.avg_pool(x_small).view(B, C)
        channel_att_2 = self.channel_attention_2(vec_y_2).view(B, C, 1, 1)
        x_small = x_small * channel_att_2 
        x = x + x_small

        shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.cbr1 = CBR(in_channels, out_channels)
        self.cbr2 = CBR(out_channels, out_channels)
        self.cbr3 = CBR(out_channels, out_channels)

    def forward(self, x):
        out1 = self.cbr1(x)
        out2 = self.cbr2(out1)
        out3 = self.cbr3(out2)
        out3  = out3 + out1 # 残差
        # out3 = F.relu(out3, inplace=True)

        return out3

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.GELU(),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.GELU(), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class Frequency_Module(nn.Module):
    def __init__(self, in_fea=[64, 128], mid_fea=128):
        super(Frequency_Module, self).__init__()
        self.relu = nn.GELU()
        self.conv2 = nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv4 = nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv5_2 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_4 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv = nn.Conv2d(mid_fea * 2, mid_fea, kernel_size=3, padding=1)
        self.rcab = RCAB(mid_fea * 2)

    def forward(self, x2, x4):
        _, _, h, w = x2.size()
        edge2_fea = self.relu(self.conv2(x2))
        edge2 = self.relu(self.conv5_2(edge2_fea))
        edge4_fea = self.relu(self.conv4(x4))
        edge4 = self.relu(self.conv5_4(edge4_fea))
        edge4 = F.interpolate(edge4, size=(h, w), mode='bilinear', align_corners=True)
        edge = torch.cat([edge2, edge4], dim=1)
        edge = self.rcab(edge)
        edge = self.conv(edge)
        return edge

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False,
                 dropout=0.1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              dilation, groups=1,
                              bias=bias)

    def forward(self, x):
        x = self.conv(x)

        return x

from einops import rearrange
class CCE(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False,mode=None):
        super(CCE, self).__init__()
        self.num_heads_1 = num_heads
        self.temperature_1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.num_heads_2 = num_heads
        self.temperature_2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv_0_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
        self.qkv1conv_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
        self.qkv3conv_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
    
        self.qkv1conv_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
        self.qkv3conv_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)

        self.project_out_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.conv1 = nn.Conv2d(2, 1, kernel_size = 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 2, kernel_size = 3, stride=1, padding=1)
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.merge_conv1x1 = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1, 1), self.relu)
        self.rcab = RCAB(dim*2)

    def forward(self, x1, x2):
        b,c,h,w = x1.shape
        q1=self.qkv1conv_1(self.qkv_0_1(x1))
        k1=self.qkv2conv_1(self.qkv_1_1(x1))
        v1=self.qkv3conv_1(self.qkv_2_1(x1))

        q2=self.qkv1conv_2(self.qkv_0_2(x2))
        k2=self.qkv2conv_2(self.qkv_1_2(x2))
        v2=self.qkv3conv_2(self.qkv_2_2(x2))

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads_1)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads_1)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads_1)

        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads_2)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads_2)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads_2)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)

        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn1 = (q1 @ k2.transpose(-2, -1)) * self.temperature_1 # q:[4, 8, 8, 7744], k.transpose(-2, -1):[4, 8, 7744, 8]
        attn1 = attn1.softmax(dim=-1) # [4, 8, 8, 8]
        out1 = (attn1 @ v2)
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads_1, h=h, w=w)
        out1 = self.project_out_1(out1)

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.temperature_2 # q:[4, 8, 8, 7744], k.transpose(-2, -1):[4, 8, 7744, 8]
        attn2 = attn2.softmax(dim=-1) # [4, 8, 8, 8]
        out2 = (attn2 @ v1)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads_2, h=h, w=w)
        out2 = self.project_out_1(out2)

        out1 = x1 + out1
        out2 = x2 + out2
        out1 = self.relu(out1)
        out2 = self.relu(out2)

        rgb_gap = torch.mean(out1, dim=1, keepdim=True)
        fre_gap = torch.mean(out2, dim=1, keepdim=True)
        stack_gap = torch.cat([rgb_gap, fre_gap], dim=1)  
        stack_gap = self.conv1(stack_gap)
        stack_gap = self.relu(stack_gap)
        stack_gap = self.conv2(stack_gap)   
        stack_gap = self.sigmoid(stack_gap)
        rgb_ = stack_gap[:, 0:1, :, :] * out1 
        fre_ = stack_gap[:, 1:2, :, :] * out2 
        merge_feature = torch.cat([rgb_, fre_], dim=1)
        merge_feature = self.rcab(merge_feature)
        merge_feature = self.merge_conv1x1(merge_feature)

        spa_out = (x1 + out1 + merge_feature) / 3
        spa_out = self.relu(spa_out)
        return spa_out

import os

if 'DWCONV_IMPL' in os.environ:
    try:
        sys.path.append(os.environ['DWCONV_IMPL'])
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        def get_dwconv(dim, kernel, bias):
            return DepthWiseConv2dImplicitGEMM(dim, kernel, bias)
        # print('Using Megvii large kernel dw conv impl')
    except:
        print(traceback.format_exc())
        def get_dwconv(dim, kernel, bias):
            return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

        # print('[fail to use Megvii Large kernel] Using PyTorch large kernel dw conv impl')
else:
    def get_dwconv(dim, kernel, bias):
            return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

    # print('Using PyTorch large kernel dw conv impl')
class FeatureDisentanglement(nn.Module):
    def __init__(self, dim, dim_2, targed_dims, order=4, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2*dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        
        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
        )

        self.scale = s

        # print('[gconv]', order, 'order with dims=', self.dims, 'scale=%.4f'%self.scale)

        self.cbr_1 = CBR(dim, dim, kernel_size=3, stride=1, dilation=1, padding=1)
        self.cbr_2 = CBR(dim_2, dim, kernel_size=3, stride=1, dilation=1, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_1 = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, dim),
            nn.Sigmoid()
        )
        self.channel_attention_2 = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, dim),
            nn.Sigmoid()
        )
        self.cbr_3 = CBR(dim, targed_dims, kernel_size=3, stride=1, dilation=1, padding=1)
    # def forward(self, x, mask=None, dummy=False):
    def forward(self, x, x_small):
        x = self.cbr_1(x)
        x_small = self.cbr_2(x_small)
        x_small = F.interpolate(x_small, size=x.size()[2:],  mode='bicubic',align_corners = False)
        B, C, _, _ = x.size()

        vec_y_1 = self.avg_pool(x).view(B, C)
        channel_att_1 = self.channel_attention_1(vec_y_1).view(B, C, 1, 1)
        x = x * channel_att_1 

        vec_y_2 = self.avg_pool(x_small).view(B, C)
        channel_att_2 = self.channel_attention_2(vec_y_2).view(B, C, 1, 1)
        x_small = x_small * channel_att_2 
        x = x + x_small
        
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order -1):
            x = self.pws[i](x) * dw_list[i+1]

        x = self.proj_out(x)

        x = self.cbr_3(x)

        return x

class IDFNet(nn.Module):
    def __init__(self):
    # def __init__(self):
        super().__init__()
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        embedding_dims=[64, 128, 320, 512]

        self.fgc3 = FGC(embedding_dims[3] // 8, embedding_dims[3] // 8, focus_background=True,
                                     opr_kernel_size=7, iterations=1)
        self.fgc2 = FGC(embedding_dims[1]//2, embedding_dims[3] // 8, focus_background=True,
                                     opr_kernel_size=7, iterations=1)
        self.fgc1 = FGC(embedding_dims[0]//2, embedding_dims[1] // 2, focus_background=True,
                                     opr_kernel_size=7,
                                     iterations=1)
        self.fgc0 = FGC(embedding_dims[0] // 4, embedding_dims[0] // 2, focus_background=False,
                                     opr_kernel_size=7, iterations=1)

        self.cbr = CBR(in_channels=embedding_dims[3], out_channels=embedding_dims[3] // 8,
                                          kernel_size=3, stride=1,
                                          dilation=1, padding=1)


        self.predict_conv = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dims[3] // 8, out_channels=1, kernel_size=3, padding=1, stride=1))
        
        self.moga0 = FeatureAggregation(embed_dims=64,embed_dims_2=128,targed_dims=32)
        self.moga1 = FeatureAggregation(embed_dims=128,embed_dims_2=320,targed_dims=64)
        self.moga2 = FeatureAggregation(embed_dims=320,embed_dims_2=512,targed_dims=64)
        self.moga3 = FeatureAggregation(embed_dims=32,embed_dims_2=64,targed_dims=16)

        self.lap_pyramid = Lap_Pyramid_Conv(num_high=5) # try2

        self.conv0_res0 = Basic_Conv(3,64,kernel_size=3,stride=2, padding=1)
        self.conv1_res0 = Basic_Conv(64,64,kernel_size=3,stride=2, padding=1)
        self.res0 = RCAB(64)

        self.conv0_res1 = Basic_Conv(3,64,kernel_size=3,stride=2, padding=1)
        self.res1 = RCAB(64)

        self.conv0_res2 = Basic_Conv(3,64,kernel_size=3,stride=1, padding=1)
        self.res2 = RCAB(64) # 对应layer[0]

        self.conv0_res3 = Basic_Conv(3,64,kernel_size=3,stride=1, padding=1)
        self.conv1_res3 = Basic_Conv(64,128,kernel_size=3,stride=1, padding=1)
        self.res3 = RCAB(128) # 对应layer[1]

        self.res_fusion = RCAB(192)
        self.conv0_res_fusion = Basic_Conv(192,64,kernel_size=3,stride=1, padding=1)

        self.conv0_res4 = Basic_Conv(3,64,kernel_size=3,stride=1, padding=1)
        self.conv1_res4 = Basic_Conv(64,320,kernel_size=3,stride=1, padding=1)
        self.res4 = RCAB(320) # 对应layer[2]

        self.fusion1 = Frequency_Module(in_fea=[64, 64], mid_fea=64)
        self.fusion2 = Frequency_Module(in_fea=[128, 128], mid_fea=128)
        self.fusion3 = Frequency_Module(in_fea=[320, 320], mid_fea=320)

        self.out_conv_evidence = BasicConv2d(64, 2, kernel_size=1)

        self.cce_fusion1 = CCE(dim=64)
    def forward(self, x):   
        layer = self.backbone(x)
        pyramid = self.lap_pyramid.pyramid_decom(x)

        pyramid[0] = self.res0(self.conv1_res0(self.conv0_res0(pyramid[0]))) # [1,64,176,176]
        pyramid[1] = self.res1(self.conv0_res1(pyramid[1])) # [1,64,88,88]
        pyramid[2] = self.res2(self.conv0_res2(pyramid[2])) 

        pyramid_fusion = torch.cat([pyramid[0], pyramid[1], pyramid[2]], dim=1)
        pyramid_fusion = self.res_fusion(pyramid_fusion)
        pyramid_fusion = self.conv0_res_fusion(pyramid_fusion)
        pyramid[2] = pyramid_fusion

        pyramid[3] = self.res3(self.conv1_res3(self.conv0_res3(pyramid[3]))) 
        pyramid[4] = self.res4(self.conv1_res4(self.conv0_res4(pyramid[4])))

        # tyr4
        # layer[0] = self.fusion1(layer[0],pyramid[2])
        layer[0] = self.cce_fusion1(layer[0],pyramid[2])
        # layer[1] = self.fusion2(layer[1],pyramid[3])
        # layer[2] = self.fusion3(layer[2],pyramid[4])

        s2 = self.moga0(layer[0], layer[1])

        s3 = self.moga1(layer[1], layer[2])

        s4 = self.moga2(layer[2], layer[3])

        s5 = self.cbr(layer[3])

        s1 = self.moga3(s2, s5)
        # s1 = self.fd3(s2, s5)
        s1 = F.interpolate(s1, scale_factor=2,  mode='bicubic',align_corners = False)


        predict4 = self.predict_conv(s5)

        fgc3, predict3 = self.fgc3(s4, s5, predict4)
        fgc2, predict2 = self.fgc2(s3, fgc3, predict3)
        fgc1, predict1 = self.fgc1(s2, fgc2, predict2)
        fgc0, predict0 = self.fgc0(s1, fgc1, predict1)
    
        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict0 = F.interpolate(predict0, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        return predict0, predict1, predict2, predict3, predict4

    def infer(self, input):
        evidence = F.softplus(input)
        return evidence

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,dilation=1):
        super(CBR, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,dilation=dilation)
        self.norm_cfg = {'type': 'BN', 'requires_grad': True}
        _, self.bn = build_norm_layer(self.norm_cfg, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)

        return x

import numpy as np
import cv2

def get_open_map(input,kernel_size,iterations):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    open_map_list = map(lambda i: cv2.dilate(i.permute(1, 2, 0).detach().numpy(), kernel=kernel, iterations=iterations), input.cpu())
    open_map_tensor = torch.from_numpy(np.array(list(open_map_list)))
    return open_map_tensor.unsqueeze(1).cuda()

class Basic_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Basic_Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AP_MP(nn.Module):
    def __init__(self,stride=2):
        super(AP_MP,self).__init__()
        self.sz=stride
        self.gapLayer=nn.AvgPool2d(kernel_size=self.sz,stride=self.sz)
        self.gmpLayer=nn.MaxPool2d(kernel_size=self.sz,stride=self.sz)

    def forward(self,x1,x2):
        B,C,H,W=x1.size()
        apimg=self.gapLayer(x1)
        mpimg=self.gmpLayer(x2)
        byimg=torch.norm(abs(apimg-mpimg),p=2,dim=1,keepdim=True)
        return byimg

class FGC(nn.Module):
    def __init__(self, channel1, channel2,focus_background = True, opr_kernel_size = 3,iterations = 1):
        super(FGC, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2
        self.focus_background = focus_background
        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.GELU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        self.uncer_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())

        #只用来查看参数
        self.increase_input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=1))
        self.increase_uncer_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=1))
        self.output_map_sal = nn.Conv2d(self.channel1, 1, 7, 1, 3)
        self.beta = nn.Parameter(torch.ones(1))


        self.conv2 = nn.Conv2d(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3, padding=1,
                               stride=1)

        self.conv_cur_dep1 = Basic_Conv(2 * self.channel1, self.channel1, 3, 1, 1)

        self.conv_cur_dep2 = Basic_Conv(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3,
                                       padding=1, stride=1)

        self.conv_cur_dep3 = Basic_Conv(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3,
                                       padding=1, stride=1)

        self.opr_kernel_size = opr_kernel_size

        self.iterations = iterations

        self.glbamp=AP_MP()

        self.conv_context_low_to_high = nn.Sequential(
            nn.Conv2d(2*channel1, channel1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel1),
            nn.GELU()
        )
        self.conv_context_high_to_low = nn.Sequential(
            nn.Conv2d(2*channel1, channel1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel1),
            nn.GELU()
        )

    def forward(self, x, small_x, in_map):
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction
        # un_map: uncertainty map

        small_x = self.up(small_x)
        # return small_x,small_x
        high_to_low = torch.cat([small_x, x], dim=1)
        low_to_high = torch.cat([x, small_x], dim=1)
        diff_1 = self.conv_context_high_to_low(high_to_low)
        diff_2 = self.conv_context_low_to_high(low_to_high)

        diff_map_1 = self.glbamp(small_x, x)
        diff_map_2 = self.glbamp(x, small_x)
        # diff = diff_1 + diff_2
        diff_map = diff_map_1 + diff_map_2

        # return diff_map

        # diff_map = self.glbamp(x,small_x)

        input_map = self.input_map(in_map)
        diff_map = self.uncer_map(diff_map)

        if self.focus_background:
            self.increase_map = self.increase_input_map(get_open_map(input_map, self.opr_kernel_size, self.iterations) - input_map)
            # b_feature = x * self.increase_map #当前层中,关注深层部分没有关注的部分
            self.difference_map = self.increase_uncer_map(get_open_map(diff_map, self.opr_kernel_size, self.iterations) - diff_map)
            b_feature = x * self.increase_map + x * self.difference_map + x
            # b_feature = x * self.increase_map + x * self.difference_map  # v1
            # b_feature = x * self.difference_map 

        else:
            # b_feature = x * input_map  #在当前层中，对深层部分关注的部分更加关注，同时也关注一下其他部分
            b_feature = x * input_map + x * diff_map + x
            # b_feature = x * input_map + x * diff_map # v1 
            # b_feature = x * diff_map 
        #b_feature = x
        fn = self.conv2(b_feature)


        refine2 = self.conv_cur_dep1(torch.cat((small_x, self.beta * fn),dim=1))
        refine2 = self.conv_cur_dep2(refine2)
        refine2 = self.conv_cur_dep3(refine2)

        output_map = self.output_map_sal(refine2)

        return refine2, output_map




if __name__ =='__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    from thop import profile
    net = SARNet('pvt_v2_b3').cuda()
    data = torch.randn(1, 3, 672, 672).cuda()
    flops, params = profile(net, (data,))
    print('flops: %.2f G, params: %.2f M' % (flops / (1024*1024*1024), params / (1024*1024)))
    y = net(data)
    for i in y:
        print(i.shape)

