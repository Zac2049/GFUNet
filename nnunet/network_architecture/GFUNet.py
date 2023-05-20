import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential
from nnunet.network_architecture.neural_network import SegmentationNetwork
from monai.networks.blocks import UnetrUpBlock, UnetrBasicBlock, UnetResBlock
from typing import Optional, Sequence, Tuple, Union

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GlobalFilter(nn.Module):
    def __init__(self, dim, size, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(size, size, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x, spatial_size=None):
        # print("before x", x.shape)
        B, N, C = x.shape
        a = b = int(math.sqrt(N))
        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        _, _, D, _ = x.shape
        # self.complex_weight = nn.Parameter(self.complex_weight[:, :D, :]* 0.02)
        weight = torch.view_as_complex(self.complex_weight[:, :D, :])
        # print("mid x", x.shape, "weight", weight.shape)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        # print("after x", x.shape)

        x = x.reshape(B, N, C)

        return x

class Block(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x

class  BlockLayerScale(nn.Module):

    def __init__(self, dim, size, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
                norm_layer=nn.LayerNorm, h=14, w=8, init_values=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, size, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * 4.)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

    def forward(self, x):
        x = x + self.drop_path(self.gamma * (self.norm2(self.filter(self.norm1(x)))))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W # add H, W


class DownLayer(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=56, dim_in=64, dim_out=128):
        super().__init__()
        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)
        self.num_patches = img_size * img_size // 4

    def forward(self, x):
#         B, N, C = x.size()
#         x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
#         x = self.proj(x).permute(0, 2, 3, 1)
#         x = x.reshape(B, -1, self.dim_out)
#         return x
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W # add H, W


class Partition(nn.Module):
    """encoder"""

    def __init__(
        self,
        in_chns: int,
        out_chns: int,
        depth: int,
        size: int,
        img_size=224,
        patch_size=2,# 4
        mlp_ratio=[4, 4, 4, 4],
        # norm: Union[str, tuple] = ("instance", {"affine": True}),
        drop_rate=0., drop_path_rate=0., norm_layer=None, init_values=0.001, no_layerscale=False, dropcls=0
    ):

        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        dpr = [drop_path_rate for _ in range(depth)]

        h = img_size // patch_size
        w = h // 2 + 1
        self.block = nn.Sequential(*[
                    BlockLayerScale(
                    dim=out_chns, size = size, mlp_ratio=mlp_ratio,
                    drop=drop_rate, drop_path=dpr[j], norm_layer=norm_layer, h=h, w=w, init_values=init_values)
                for j in range(depth)
        ])
        self.norm = norm_layer(out_chns)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chns, embed_dim=out_chns)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, out_chns))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.dwconv = nn.Conv2d(in_chns, out_chns, kernel_size=7, padding=3, groups=1)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x, H, W = self.patch_embed(x)
        x = x + self.pos_embed# i == 0
        x = self.pos_drop(x)

        # block cant change image size
        x = self.block(x)
        x = self.norm(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # x = x.permute(0, 2, 1)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

class Down(nn.Module):
    """encoder"""

    def __init__(
        self,
        in_chns: int,
        out_chns: int,
        depth: int,
        size: int,
        img_size=224,
        patch_size=4,
        mlp_ratio=[4, 4, 4, 4],
        # norm: Union[str, tuple] = ("instance", {"affine": True}),
        drop_rate=0., drop_path_rate=0., norm_layer=None, init_values=0.001, no_layerscale=False, dropcls=0
    ):

        super().__init__()
        embed_dim = out_chns
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        dpr = [drop_path_rate for _ in range(depth)]

        h = img_size // patch_size
        w = h // 2 + 1
        self.block = nn.Sequential(*[
                    BlockLayerScale(
                    dim=embed_dim, size=size, mlp_ratio=mlp_ratio,
                    drop=drop_rate, drop_path=dpr[j], norm_layer=norm_layer, h=h, w=w, init_values=init_values)
                for j in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patch_embed = DownLayer(size, in_chns, out_chns)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        # x = x.permute(0, 2, 1)
        x = self.pos_drop(x)
        x, H, W = self.patch_embed(x)
        
        # block cant change image size
        x = self.block(x)
        x = self.norm(x)
        # x = x.permute(0, 2, 1)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

class Down_non(nn.Module):
    """encoder"""

    def __init__(
        self,
        in_chns: int,
        out_chns: int,
        depth: int,
        size: int,
        img_size=224,
        patch_size=4,
        mlp_ratio=[4, 4, 4, 4],
        # norm: Union[str, tuple] = ("instance", {"affine": True}),
        drop_rate=0., drop_path_rate=0., norm_layer=None, init_values=0.001, no_layerscale=False, dropcls=0
    ):

        super().__init__()
        embed_dim = out_chns
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        dpr = [drop_path_rate for _ in range(depth)]

        h = img_size // patch_size
        w = h // 2 + 1
        self.dwconv = nn.Conv2d(embed_dim, embed_dim, kernel_size=7, padding=3, groups=embed_dim)
        # self.block = nn.Sequential(*[
        #             BlockLayerScale(
        #             dim=embed_dim, size=size, mlp_ratio=mlp_ratio,
        #             drop=drop_rate, drop_path=dpr[j], norm_layer=norm_layer, h=h, w=w, init_values=init_values)
        #         for j in range(depth)
        # ])
        self.norm = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patch_embed = DownLayer(size, in_chns, out_chns)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        # x = x.permute(0, 2, 1)
        x = self.pos_drop(x)
        x, H, W = self.patch_embed(x)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.dwconv(x)
        # x = self.norm(x)
        # x = x.permute(0, 2, 1)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x
        
# class UpCat(nn.Module):
#     """upsampling, concatenation with the encoder feature map"""

#     def __init__(
#         self,
#         in_chns: int,
#         out_chns: int,
#         h: int,
#         spatial_dims = 2,
#         dropout: float = 0.0,
#         upsample: str = "deconv",
#         # pre_conv: Optional[Union[nn.Module, str]] = "default",
#         interp_mode: str = "bilinear",
#         # align_corners: Optional[bool] = True,
#         # halves: bool = True,
#         # is_pad: bool = True,
#     ):
        
#         super().__init__()
#         norm_layer = partial(nn.LayerNorm, eps=1e-6)
#         self.norm = nn.LayerNorm([in_chns, h, h])
#         self.deconv = nn.Conv2d(in_chns, out_chns, 3, stride=1, padding=1)
#         self.conv = nn.Conv2d(2*out_chns, out_chns, 3, stride=1, padding=1)
#         # self.res_block = UnetResBlock(spatial_dims, in_chns, out_chns, kernel_size=3, stride=1, norm_name = "instance",)
#         # 之后考虑使用
        

#     def forward(self, x: torch.Tensor, x_e: torch.Tensor):
#         x = F.relu(F.interpolate(self.norm(self.deconv(x)),scale_factor=(2,2),mode ='bilinear'))
#         x = torch.cat([x, x_e], dim=1)
#         x = self.conv(x)
#         return x


# embed_dim根据网络size决定
class GFUNet(SegmentationNetwork):
    
    def __init__(
        self, 
        num_classes=4, 
        img_size=224, 
        patch_size=2, # 4
        feat_size=[32, 64, 128, 256, 512], # modifiable
        depth=[1,1,1,1,1], # modifiable
        mlp_ratio=[4, 4, 4, 4],
        drop_rate=0., drop_path_rate=0., norm_layer=None, init_values=0.001, no_layerscale=False, norm_name = "instance",dropcls=0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.do_ds=False
        self.num_classes = num_classes
        res_block = True
        in_chan = 3
        # original
        self.partition_down = Partition(in_chan, feat_size[0], depth[0], img_size//2)# 3 32 112
        self.down_2 = Down(feat_size[0], feat_size[1], depth[1], img_size//4)# 32 64 56
        self.down_3 = Down(feat_size[1], feat_size[2], depth[2], img_size//8)# 64 128
        self.down_4 = Down(feat_size[2], feat_size[3], depth[3], img_size//16)# 128 256
        self.down_5 = Down(feat_size[3], feat_size[4], depth[4], img_size//32)

        self.upcat_5 = UnetrUpBlock(2, feat_size[4], feat_size[3], kernel_size=3, upsample_kernel_size=2, norm_name = norm_name, res_block=res_block)
        self.upcat_4 = UnetrUpBlock(2, feat_size[3], feat_size[2], kernel_size=3, upsample_kernel_size=2, norm_name = norm_name, res_block=res_block)
        self.upcat_3 = UnetrUpBlock(2, feat_size[2], feat_size[1], kernel_size=3, upsample_kernel_size=2, norm_name = norm_name, res_block=res_block)
        self.upcat_2 = UnetrUpBlock(2, feat_size[1], feat_size[0], kernel_size=3, upsample_kernel_size=2, norm_name = norm_name, res_block=res_block)
        self.upcat_1 = UnetrBasicBlock(2, feat_size[0], feat_size[0], kernel_size=3, stride=1, norm_name = norm_name, res_block=res_block)
        
        self.final = nn.Conv2d(feat_size[0], num_classes, kernel_size=1)
        # self.final=[]
        # for i in range(len(self.depths)-1):
        #     self.final.append(final_patch_expanding(self.embed_dim*2**i,self.num_classes,patch_size=self.patch_size))
        # self.final=nn.ModuleList(self.final)
        
        # abandoned decoder
        # self.upcat_5 = UpCat(embed_dim[4], embed_dim[3], h[0], depth[4])
        # self.upcat_4 = UpCat(embed_dim[3], embed_dim[2], h[1], depth[3])
        # self.upcat_3 = UpCat(embed_dim[2], embed_dim[1], h[2], depth[2])
        # self.upcat_2 = UpCat(embed_dim[1], embed_dim[0], h[3], depth[1])
        # self.upcat_1 = UpCat(embed_dim[0], in_chan, h[4], depth[0])

        # tested encoder
        # self.partition_down = nn.Conv2d(in_chan, feat_size[0], 3, stride=1, padding=1)# 3 32 112
        # self.down_2 = nn.Conv2d(feat_size[0], feat_size[1], 3, stride=1, padding=1)# 32 64 56
        # self.down_3 = nn.Conv2d(feat_size[1], feat_size[2], 3, stride=1, padding=1)# 64 128
        # self.down_4 = nn.Conv2d(feat_size[2], feat_size[3], 3, stride=1, padding=1)# 128 256
        # self.down_5 = nn.Conv2d(feat_size[3], feat_size[4], 3, stride=1, padding=1)
        # self.norm1 = nn.LayerNorm(feat_size[0])
        # self.norm2 = nn.LayerNorm(feat_size[1])
        # self.norm3 = nn.LayerNorm(feat_size[2])
        # self.norm4 = nn.LayerNorm(feat_size[3])
        # self.norm5 = nn.LayerNorm(feat_size[4])
        # # tested decoder
        # self.upcat_5 = nn.Conv2d(feat_size[4], feat_size[3], 3, stride=1, padding=1 )
        # self.upcat_4 = nn.Conv2d(feat_size[3], feat_size[2], 3, stride=1, padding=1 )
        # self.upcat_3 = nn.Conv2d(feat_size[2], feat_size[1], 3, stride=1, padding=1 )
        # self.upcat_2 = nn.Conv2d(feat_size[1], feat_size[0], 3, stride=1, padding=1 )
        # self.upcat_1 = nn.Conv2d(feat_size[0], feat_size[0], 3, stride=1, padding=1)
        # self.final = nn.Conv2d(feat_size[0], num_classes, kernel_size=1)
        # inchan 3 or 4 related
        
    def expand(self, x):
        B, C, D = x.shape[0], x.shape[1], x.shape[2]
        H = int(math.sqrt(D))
        x = x.reshape(B, C, H, H).contiguous()
        

    def forward(self, x):
        
        x = x.repeat(1, 3, 1, 1)# grayscale to RGB only for ACDC
#         print("x", x.shape)
        x1 = self.partition_down(x)
        # print("x1", x1.shape)
        x2 = self.down_2(x1)
        # print("x2", x2.shape)
        x3 = self.down_3(x2)
        # print("x3", x3.shape)
        x4 = self.down_4(x3)
        # print("x4", x4.shape)
        x5 = self.down_5(x4)
        u4 = self.upcat_5(x5, x4)
        u3 = self.upcat_4(u4, x3)
        u2 = self.upcat_3(u3, x2)
        u1 = self.upcat_2(u2, x1)
        u = self.upcat_1(u1)
        # have not added norm
        u = F.relu(F.interpolate(u, scale_factor=(2,2),mode ='bilinear'))
        out = self.final(u)
        return out
        # deep supervision
        # seg_outputs=[]
        # for i in range(len(u)):  
        #     seg_outputs.append(self.final[-(i+1)](u[i]))
        # if self.do_ds:
        #     # for training
        #     return seg_outputs[::-1]
        #     #size [[224,224],[112,112],[56,56]]

        # else:
        #     #for validation and testing
        #     return seg_outputs[-1]
        #     #size [[224,224]]

        # x5 torch.Size([8, 512, 7, 7])
        # u2 torch.Size([8, 64, 56, 56])
        # u1 torch.Size([8, 32, 112, 112])
        # u torch.Size([8, 32, 112, 112])
        # out torch.Size([8, 4, 112, 112])

        
        # x = x.repeat(1, 3, 1, 1)# grayscale to RGB
        # B = x.shape[0]
        # # print("x", x.shape)
        # x1 = F.relu(F.max_pool2d((self.partition_down(x)), 2, 2))
        # # print("x1", x1.shape)
        # x2 = F.relu(F.max_pool2d((self.down_2(x1)), 2, 2))
        # # print("x2", x2.shape)
        # x3 = F.relu(F.max_pool2d((self.down_3(x2)), 2, 2))
        # # print("x3", x3.shape)
        # x4 = F.relu(F.max_pool2d((self.down_4(x3)), 2, 2))
        # # print("x4", x4.shape)
        # x5 = F.relu(F.max_pool2d((self.down_5(x4)), 2, 2))

        # u4 = F.relu(F.interpolate((self.upcat_5(x5)), scale_factor=(2,2),mode='bilinear'))
        # u3 = F.relu(F.interpolate((self.upcat_4(u4)), scale_factor=(2,2),mode='bilinear'))
        # u2 = F.relu(F.interpolate((self.upcat_3(u3)), scale_factor=(2,2),mode='bilinear'))
        # u1 = F.relu(F.interpolate((self.upcat_2(u2)), scale_factor=(2,2),mode='bilinear'))
        # u = F.relu(F.interpolate((self.upcat_1(u1)), scale_factor=(2,2),mode='bilinear'))


