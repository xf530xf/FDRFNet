import torch
import torch.nn as nn
import torch.nn.functional as F
from laplaciannet import RFAE
Act = nn.ReLU
from DeiT import deit_base_distilled_patch16_384
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint
from einops import rearrange
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn
    def initialize(self):
        weight_init(self)


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

    def initialize(self):
        weight_init(self)


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

    def initialize(self):
        weight_init(self)

class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

    def initialize(self):
        weight_init(self)

def weight_init(module):
    for n, m in module.named_children():
      #  print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d,nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Sequential,nn.ModuleList,nn.ModuleDict)):
            weight_init(m)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (LayerNorm,nn.ReLU,Act,nn.AdaptiveAvgPool2d,nn.Softmax,nn.AvgPool2d, nn.Sigmoid)):
            pass
        else:
            m.initialize()

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )

    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    def initialize(self):
        weight_init(self)
        
def window_partition(x,window_size):
    # input B C H W
    x = x.permute(0,2,3,1)
    B,H,W,C = x.shape
    x = x.view(B,H//window_size,window_size,W//window_size,window_size,C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1,window_size,window_size,C)
    return windows #B_ H_ W_ C

def window_reverse(windows,window_size,H,W):
    B=int(windows.shape[0]/(H*W/window_size/window_size))
    x = windows.view(B,H//window_size,W//window_size,window_size,window_size,-1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B,H,W,-1)
    return x.permute(0,3,1,2)

class MLP(nn.Module):
    def __init__(self, inchannel,outchannel, bias=False):
        super(MLP, self).__init__()
        self.conv1 = nn.Linear(inchannel, outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.ln   = nn.LayerNorm(outchannel)
        

    def forward(self, x):
        return self.relu(self.ln(self.conv1(x))+x)
    def initialize(self):
        weight_init(self)


class WAttention(nn.Module): # x hf  y  lf
    def __init__(self, dim, num_heads=8, level=8,qkv_bias=True, qk_scale=None):
        super().__init__()
        self.level = level
        self.mul = nn.Sequential(ConvBNReLu(dim,dim),ConvBNReLu(dim,dim,kernel_size=1,padding=0))
        self.add = nn.Sequential(ConvBNReLu(dim,dim),ConvBNReLu(dim,dim,kernel_size=1,padding=0))

        self.conv_x = nn.Sequential(ConvBNReLu(dim,dim),ConvBNReLu(dim,dim,kernel_size=1,padding=0))

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)
        
        self.lnx = nn.LayerNorm(dim)
        self.lny = nn.LayerNorm(dim)
        self.ln = nn.LayerNorm(dim)
        
        self.shortcut = nn.Linear(dim,dim)

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3, stride=1, padding=1),
            LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim,dim,kernel_size=1, stride=1, padding=0),
            LayerNorm(dim)
        )
        self.mlp = MLP(dim,dim)


    def forward(self, x, y): 
        origin_size = x.shape[2]
        ws = origin_size//self.level//4 
        y = F.interpolate(y,size=x.size()[2:], mode='bilinear', align_corners=True)
        x = self.conv_x(x)

        x = window_partition(x,ws) 
        y = window_partition(y,ws)

        x = x.view(x.shape[0], -1, x.shape[3])
        sc1 = x
        x = self.lnx(x)
        y = y.view(y.shape[0], -1, y.shape[3])
        y = self.lny(y)
        B, N, C = x.shape
        y_kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q= self.q(x).reshape(B,N,1,self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q      = x_q[0]   
        y_k, y_v = y_kv[0],y_kv[1]
        attn = (x_q @ y_k.transpose(-2, -1)) * self.scale # B_ C WW WW
        attn = attn.softmax(dim=-1)
        x = (attn @ y_v).transpose(1, 2).reshape(B, N, C) # B' N C
        x = self.act(x+sc1)
        x = self.act(x+self.mlp(x))
        x = x.view(-1,ws,ws,C)
        x = window_reverse(x,ws,origin_size,origin_size) # B C H W
        x = self.act(self.conv2(x)+x)
        return x
        
    def initialize(self):
        weight_init(self)


class DB1(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(DB1,self).__init__()
        self.squeeze2 = nn.Sequential(nn.Conv2d(outplanes, outplanes, kernel_size=3,stride=1,dilation=2,padding=2), LayerNorm(outplanes), nn.ReLU(inplace=True))
        self.squeeze1 = nn.Sequential(nn.Conv2d(inplanes, outplanes,kernel_size=1,stride=1,padding=0), LayerNorm(outplanes), nn.ReLU(inplace=True))
        self.relu = Act(inplace=True)

    def forward(self, x,z):
        if(z is not None and z.size()!=x.size()):
            z = F.interpolate(z, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = self.squeeze1(x)
        z = x+self.squeeze2(x) if z is None else x+self.squeeze2(x+z)       
        return z,z

    def initialize(self):
        weight_init(self)

class DB2(nn.Module):
    def __init__(self,inplanesx,inplanesz,outplanes,head=8):
        super(DB2,self).__init__()
        self.inplanesx=inplanesx
        self.inplanesz=inplanesz

        self.short_cut = nn.Conv2d(inplanesz, outplanes, kernel_size=1, stride=1, padding=0)
        self.conv=(nn.Sequential(
            nn.Conv2d((inplanesx+inplanesz),outplanes,kernel_size=3, stride=1, padding=1),
            LayerNorm(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            LayerNorm(outplanes),
            nn.ReLU(inplace=True)
        ))
        self.conv2 = nn.Sequential(
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            LayerNorm(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            LayerNorm(outplanes),
            nn.ReLU(inplace=True)
        )

    def forward(self,x,z):
        z = F.interpolate(z, size=x.size()[2:], mode='bilinear')
        p = self.conv(torch.cat((x,z),dim=1))
        sc = self.short_cut(z)
        p  = p+sc
        p2 = self.conv2(p)
        p  = p+p2
        return p,p
    
    def initialize(self):
        weight_init(self)


class ConvBNReLu(nn.Module):
    def __init__(self,inplanes,outplanes,kernel_size=3,dilation=1,padding=1):
        super(ConvBNReLu,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes,outplanes,kernel_size=kernel_size, dilation=dilation,stride=1, padding=padding),
            LayerNorm(outplanes),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)

    def initialize(self):
        weight_init(self)

class FuseLayer(nn.Module):
    def __init__(self):
        super(FuseLayer,self).__init__()
        self.fuse_16 = WAttention(384,num_heads=8,level=1)
        self.fuse_8  = WAttention(192,num_heads=8,level=2)
        self.fuse_4  = WAttention(96,num_heads=8,level=4)

        self.sqz0     = nn.Sequential(ConvBNReLu(768,384),ConvBNReLu(384,384,kernel_size=1,padding=0))
        self.sqz_16  = DB1(768,384)
        self.sqz1     = nn.Sequential(ConvBNReLu(384,192),ConvBNReLu(192,192,kernel_size=1,padding=0))
        self.sqz_8   = DB1(768,192)
        self.sqz2     = nn.Sequential(ConvBNReLu(192,96),ConvBNReLu(96,96,kernel_size=1,padding=0))
        self.sqz_4   = DB1(768,96)

        self.R1  = WAttention(192,num_heads=8,level=2)
        self.R2  = WAttention(96,num_heads=8,level=4)
        self.R3  = WAttention(96,num_heads=8,level=4)

        self.block_16 = CGAFusion(384)
        self.block_8 = CGAFusion(192)
        self.block_4 = CGAFusion(96)
        self.block_r1 = CGAFusion(192)
        self.block_r2 = CGAFusion(96)
        self.block_r3 = CGAFusion(96)
        
    def forward(self,lap_out,vit_out):  # 512 128 32 768     
        vit_16  = self.sqz0(vit_out[11])
        
        # vit_16  = self.sqz0(vit_out)
        vit_8 = self.sqz1(vit_16)
        vit_4  = self.sqz2(vit_8)

# cross fusion
        vit_16 = F.interpolate(vit_16,size=lap_out['out1_16'].size()[2:], mode='bilinear', align_corners=True)      
        # fuse_16 = self.fuse_16(lap_out['out1_16'],vit_16)
        fuse_16 = self.block_16(lap_out['out1_16'],vit_16)

        vit_8 = F.interpolate(vit_8,size=lap_out['out2_8'].size()[2:], mode='bilinear', align_corners=True)     
        fuse_8 = self.block_8(lap_out['out2_8'],vit_8)

        vit_4 = F.interpolate(vit_4,size=lap_out['out3_4'].size()[2:], mode='bilinear', align_corners=True) 
        fuse_4 = self.block_4(lap_out['out3_4'],vit_4)
        
# up fusion
        fuse_8 = F.interpolate(fuse_8,size=lap_out['out1_8'].size()[2:], mode='bilinear', align_corners=True)
        fuse_8 = self.block_r1(lap_out['out1_8'],fuse_8)
        
        fuse_4 = F.interpolate(fuse_4,size=lap_out['out2_4'].size()[2:], mode='bilinear', align_corners=True)
        fuse_4= self.block_r2(lap_out['out2_4'],fuse_4)
        
        fuse_4 = F.interpolate(fuse_4,size=lap_out['out1_4'].size()[2:], mode='bilinear', align_corners=True)
        fuse_4 = self.block_r3(lap_out['out1_4'],fuse_4)
        
        return fuse_16,fuse_8,fuse_4 
    def initialize(self):
        weight_init(self)

class decoder(nn.Module):
    def __init__(self):
        super(decoder,self).__init__()
        self.fuse0 = DB1(768,384)
        self.fuse1 = DB2(384,384,192)
        self.fuse2 = DB2(192,192,96)
        self.fuse3 = DB2(96,96,48)

    def forward(self,vitf,fuse_16,fuse_8,fuse_4):

        vitf,z = self.fuse0(vitf,None)
        out_16,z = self.fuse1(fuse_16,z)
        out_8,z = self.fuse2(fuse_8,z)
        out_4,z = self.fuse3(fuse_4,z)

        return vitf,out_16,out_8,out_4
    def initialize(self):
        weight_init(self)

class re_head(nn.Module):
    def __init__(self):
        super(re_head,self).__init__()
        self.head_4= nn.Sequential(ConvBNReLu(48,48),nn.Conv2d(48,3,3,1,0))
        self.head_8= nn.Sequential(ConvBNReLu(96,48),nn.Conv2d(48,3,3,1,0))
        self.head_16= nn.Sequential(ConvBNReLu(384,48),nn.Conv2d(48,3,3,1,0))
    def forward(self,fuse_4,fuse_8,fuse_16):
        # print(fuse_4.shape, fuse_8.shape, fuse_16.shape)
        re_4 = self.head_4(fuse_4)
        re_8 = self.head_8(fuse_8)
        re_16 = self.head_16(fuse_16)

        return re_4, re_8,re_16
    def initialize(self):
        weight_init(self)

class FDRFNet(nn.Module):
    def __init__(self, cfg=None):
        super(FDRFNet, self).__init__()
        self.cfg      = cfg
        self.linear1 = nn.Conv2d(48, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(96, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(192, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(384, 1, kernel_size=3, stride=1, padding=1)

        self.fuselayer = FuseLayer()
        self.decoder = decoder()
        self.re_decoder = decoder()
        self.re_head = re_head()

        self.RFAE = RFAE()

        self.qformer_conv = nn.Conv2d(144, 144*3, kernel_size=1)
        if self.cfg is None or self.cfg.snapshot is None:
            weight_init(self)

        self.bkbone   = deit_base_distilled_patch16_384()
        self.qformer = QFormer(img_size=224,
                patch_size=4,  #4
                in_chans=3,
                num_classes=1000,
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[ 4, 8, 16, 32 ],     #     [ 4, 8, 16, 32 ]
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                drop_path_rate=0.1,          
                ape=False,
                patch_norm=True,
                use_checkpoint=False,
                coords_lambda=0.0,
                rpe='v1',
                )
        checkpoint = torch.load('/data3/YG/FRINet/code/pre/QFormer_B_patch4_window7_224.pth', map_location='cpu')
        msg = self.qformer.load_state_dict(checkpoint['model'], strict=False)

        
        if self.cfg is not None and self.cfg.snapshot:
            print('load checkpoint')
            pretrain=torch.load(self.cfg.snapshot)
            new_state_dict = {}
            for k,v in pretrain.items():
                new_state_dict[k[7:]] = v  
            self.load_state_dict(new_state_dict, strict=True)

    def forward(self, img,RFC,shape=None,mask=None):
        shape = img.size()[2:] if shape is None else shape
        img = F.interpolate(img, size=(384,384), mode='bilinear',align_corners=True)
        # print(f'img:{img.shape}') # img:torch.Size([4, 3, 384, 384])
        vitf = self.bkbone(img) # Low Frequency Representation
        # vitf = self.qformer(img)
        # vitf = vitf.unsqueeze(3)
        # vitf = self.qformer_conv(vitf)
        # vitf = vitf.view(vitf.shape[0], 768, 24, 24)
        
        # print(f'vitf:{vitf[11].shape}') # len=12 # 均为torch.Size([4, 768, 24, 24])
        # for item in vitf:
        #     print(item.shape)
        cnn_out = self.RFAE(RFC) #High Frequency Representation Array
        fuse_16,fuse_8,fuse_4 = self.fuselayer(cnn_out,vitf) # Progressive Frequency Representation Integration
        vitout16,out_16,out_8,out_4 = self.decoder(vitf[11],fuse_16,fuse_8,fuse_4)  # COD decoder
        # vitout16,out_16,out_8,out_4 = self.decoder(vitf,fuse_16,fuse_8,fuse_4)  # COD decoder
        re_16 ,_,re_8 ,re_4  = self.re_decoder(vitf[11],cnn_out['out1_16'],cnn_out['out2_8'],cnn_out['out3_4']) # Reconstruct decoder
        # re_16 ,_,re_8 ,re_4  = self.re_decoder(vitf,cnn_out['out1_16'],cnn_out['out2_8'],cnn_out['out3_4']) # Reconstruct decoder
        
        pred1 = F.interpolate(self.linear1(out_4), size=shape, mode='bilinear')
        pred2 = F.interpolate(self.linear2(out_8), size=shape, mode='bilinear')
        pred3 = F.interpolate(self.linear3(out_16), size=shape, mode='bilinear')
        pred4 = F.interpolate(self.linear4(vitout16), size=shape, mode='bilinear')

        re_4,re_8,re_16 = self.re_head(re_4,re_8,re_16)

        return pred1,pred2,pred3,pred4, re_4, re_8,re_16

class QFormer(nn.Module):
    r""" Vision Transformer with Quadrangle Attention
        A PyTorch impl of : `Vision Transformer with Quadrangle Attention: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/abs/2303.15105
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each QFormer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, rpe='v1', coords_lambda=0, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               rpe=rpe,
                               coords_lambda=coords_lambda)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self.apply(self.reset_parameters)

    def reset_parameters(self, m):
        if hasattr(m, '_reset_parameters'):
            m._reset_parameters()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        feature_list = []
        x, h, w = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for index, layer in enumerate(self.layers):
            # print(index)
            layer.H = h
            layer.W = w
            x = layer(x)
            h = h // 2
            w = w // 2
            # print(x.shape)
            # print(f'------{x.shape}')
            # if index == 0:
            #     y = x.view(x.size(0), 768, 24, -1)
            #     y = y[:, :, :, :24]
            # if index == 1:
            #     y = x.view(x.size(0), 768, 24, -1)
            #     padding = (24 - 16) // 2
            #     # 在最后一维两侧进行对称填充
            #     y = F.pad(y, (padding, padding, 0, 0, 0, 0, 0, 0))
            # if index == 2 or index == 3:
            #     y = x.view(x.size(0), 768, 24, -1)
            #     padding = (24 - 8) // 2
            #     y = F.pad(y, (padding, padding, 0, 0, 0, 0, 0, 0))
                # print(f'2{x.shape}')
            feature_list.append(x)

        # x = self.norm(x)  # B L C
        
        # x = self.avgpool(x.transpose(1, 2))  # B C 1 # # guang
        
        # x = x.transpose(1, 2)
        
        # x = torch.flatten(x, 1) # guang
        # return x
        # feature_list = feature_list * 3
        return feature_list

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x) # guang
        return x[3]

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        # for i, layer in enumerate(self.layers):
        #     flops += layer.flops()
        #     print(f'flops for layer {i}: {layer.flops()}')
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # B Ph*Pw C
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, H, W

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class BasicLayer(nn.Module):
    """ A basic QFormer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 rpe='v2', coords_lambda=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 rpe=rpe,
                                 coords_lambda=coords_lambda)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            blk.H = self.H
            blk.W = self.W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            self.downsample.H = self.H
            self.downsample.W = self.W
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class SwinTransformerBlock(nn.Module):
    r""" Vision Transformer with Quadrangle Attention Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, rpe='v2', coords_lambda=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.pos = nn.Conv2d(dim, dim, window_size//2*2+1, 1, window_size//2, groups=dim, bias=True)
        self.attn = QuadrangleAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, window_size=window_size, rpe=rpe, coords_lambda=coords_lambda
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x + self.pos(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).flatten(1, 2)
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, H, W)
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += self.attn.flops()

        # add local connection for our model calculation
        flops += self.dim * H * W * 7 * 7 * 4

        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class QuadrangleAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., window_size=7, rpe='v2', coords_lambda=20):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.window_num = 1
        self.coords_lambda = coords_lambda

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.identity = nn.Identity()  # for hook
        self.identity_attn = nn.Identity()  # for hook
        self.identity_distance = nn.Identity()

        self.transform = nn.Sequential(
                nn.AvgPool2d(kernel_size=window_size, stride=window_size), 
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads*9, kernel_size=1, stride=1)
            )

        self.rpe = rpe
        if rpe == 'v1':
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((window_size * 2 - 1) * (window_size * 2 - 1), num_heads))  # (2*Wh-1 * 2*Ww-1 + 1, nH) 
            # self.relative_position_bias = torch.zeros(1, num_heads) # the extra is for the token outside windows

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)
            # print('The v1 relative_pos_embedding is used')

        elif rpe == 'v2':
            q_size = window_size
            rel_sp_dim = 2 * q_size - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            trunc_normal_(self.rel_pos_h, std=.02)
            trunc_normal_(self.rel_pos_w, std=.02)
            # print('The v2 relative_pos_embedding is used')

    def forward(self, x, h, w):
        b, N, C = x.shape
        x = x.reshape(b, h, w, C).permute(0, 3, 1, 2)
        shortcut = x
        qkv_shortcut = F.conv2d(shortcut, self.qkv.weight.unsqueeze(-1).unsqueeze(-1), bias=self.qkv.bias, stride=1)
        ws = self.window_size
        padding_t = 0
        padding_d = (ws - h % ws) % ws
        padding_l = 0
        padding_r = (ws - w % ws) % ws
        expand_h, expand_w = h+padding_t+padding_d, w+padding_l+padding_r
        window_num_h = expand_h // ws
        window_num_w = expand_w // ws
        assert expand_h % ws == 0
        assert expand_w % ws == 0
        image_reference_h = torch.linspace(-1, 1, expand_h).to(x.device)
        image_reference_w = torch.linspace(-1, 1, expand_w).to(x.device)
        image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h), 0).permute(0, 2, 1).unsqueeze(0) # 2, h, w
        window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=ws)
        image_reference = image_reference.reshape(1, 2, window_num_h, ws, window_num_w, ws)
        window_center_coords = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)

        base_coords_h = torch.arange(ws).to(x.device) * 2 / (expand_h-1)
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = torch.arange(ws).to(x.device) * 2 / (expand_w-1)
        base_coords_w = (base_coords_w - base_coords_w.mean())


        expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)
        assert expanded_base_coords_h.shape[0] == window_num_h
        assert expanded_base_coords_h.shape[1] == ws
        expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)
        assert expanded_base_coords_w.shape[0] == window_num_w
        assert expanded_base_coords_w.shape[1] == ws
        expanded_base_coords_h = expanded_base_coords_h.reshape(-1)
        expanded_base_coords_w = expanded_base_coords_w.reshape(-1)
        window_coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2, 1).reshape(1, 2, window_num_h, ws, window_num_w, ws).permute(0, 2, 4, 1, 3, 5)
        # base_coords = image_reference

        qkv = qkv_shortcut
        qkv = torch.nn.functional.pad(qkv, (padding_l, padding_r, padding_t, padding_d))
        qkv = rearrange(qkv, 'b (num h dim) hh ww -> num (b h) dim hh ww', h=self.num_heads//self.window_num, num=3, dim=self.dim//self.num_heads, b=b, hh=expand_h, ww=expand_w)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if h > ws or w > ws:
            # getting the learned params for the varied windows and the coordinates of each pixel
            x = torch.nn.functional.pad(shortcut, (padding_l, padding_r, padding_t, padding_d))
            sampling_ = self.transform(x).reshape(b*self.num_heads//self.window_num, 9, window_num_h, window_num_w).permute(0, 2, 3, 1)
            sampling_offsets = sampling_[..., :2,]
            sampling_offsets[..., 0] = sampling_offsets[..., 0] / (expand_w // ws)
            sampling_offsets[..., 1] = sampling_offsets[..., 1] / (expand_h // ws)
            # sampling_offsets = sampling_offsets.permute(0, 3, 1, 2)
            sampling_offsets = sampling_offsets.reshape(-1, window_num_h, window_num_w, 2, 1)
            sampling_scales = sampling_[..., 2:4] + 1
            sampling_shear = sampling_[..., 4:6]
            sampling_projc = sampling_[..., 6:8]
            sampling_rotation = sampling_[..., -1]
            zero_vector = torch.zeros(b*self.num_heads//self.window_num, window_num_h, window_num_w).cuda()
            sampling_projc = torch.cat([
                sampling_projc.reshape(-1, window_num_h, window_num_w, 1, 2),
                torch.ones_like(zero_vector).cuda().reshape(-1, window_num_h, window_num_w, 1, 1)
                ], dim=-1)

            shear_matrix = torch.stack([
                torch.ones_like(zero_vector).cuda(),
                sampling_shear[..., 0],
                sampling_shear[..., 1],
                torch.ones_like(zero_vector).cuda()], dim=-1).reshape(-1, window_num_h, window_num_w, 2, 2)
            scales_matrix = torch.stack([
                sampling_scales[..., 0],
                torch.zeros_like(zero_vector).cuda(),
                torch.zeros_like(zero_vector).cuda(),
                sampling_scales[..., 1],
            ], dim=-1).reshape(-1, window_num_h, window_num_w, 2, 2)
            rotation_matrix = torch.stack([
                sampling_rotation.cos(),
                sampling_rotation.sin(),
                -sampling_rotation.sin(),
                sampling_rotation.cos()
            ], dim=-1).reshape(-1, window_num_h, window_num_w, 2, 2)
            basic_transform_matrix = rotation_matrix @ shear_matrix @ scales_matrix
            affine_matrix = torch.cat(
                (torch.cat((basic_transform_matrix, sampling_offsets), dim=-1), sampling_projc), dim=-2)
            window_coords_pers = torch.cat([
                window_coords.flatten(-2, -1), torch.ones(1, window_num_h, window_num_w, 1, ws*ws).cuda()
            ], dim=-2)
            transform_window_coords = affine_matrix @ window_coords_pers
            # transform_window_coords = rotation_matrix @ shear_matrix @ scales_matrix @ window_coords.flatten(-2, -1)
            _transform_window_coords3 = transform_window_coords[..., -1, :]
            _transform_window_coords3[_transform_window_coords3==0] = 1e-6
            transform_window_coords = transform_window_coords[..., :2, :] / _transform_window_coords3.unsqueeze(dim=-2)
            # _transform_window_coords0 = transform_window_coords[..., 0, :] / _transform_window_coords3
            # _transform_window_coords1 = transform_window_coords[..., 1, :] / _transform_window_coords3
            # transform_window_coords = torch.stack((_transform_window_coords0, _transform_window_coords1), dim=-2)
            # transform_window_coords = transform_window_coords[..., :2, :]
            transform_window_coords_distance = transform_window_coords.reshape(-1, window_num_h, window_num_w, 2, ws*ws, 1)
            transform_window_coords_distance = transform_window_coords_distance - window_coords.reshape(-1, window_num_h, window_num_w, 2, 1, ws*ws)
            transform_window_coords_distance = torch.sqrt((transform_window_coords_distance[..., 0, :, :]*(expand_w-1)/2) ** 2 + (transform_window_coords_distance[..., 1, :, :]*(expand_h-1)/2) ** 2)
            transform_window_coords_distance = rearrange(transform_window_coords_distance, '(b h) hh ww n1 n2 -> (b hh ww) h n1 n2', b=b, h=self.num_heads, hh=window_num_h, ww=window_num_w, n1=ws*ws, n2=ws*ws)
            transform_window_coords = transform_window_coords.reshape(-1, window_num_h, window_num_w, 2, ws, ws).permute(0, 3, 1, 4, 2, 5)
            #TODO: adjust the order of transformation

            coords = window_center_coords.repeat(b*self.num_heads, 1, 1, 1, 1, 1) + transform_window_coords

            # coords = base_coords.repeat(b*self.num_heads//self.window_num, 1, 1, 1, 1, 1) + window_coords * sampling_scales[:, :, :, None, :, None] + sampling_offsets[:, :, :, None, :, None]
            sample_coords = coords.permute(0, 2, 3, 4, 5, 1).reshape(b*self.num_heads, ws*window_num_h, ws*window_num_w, 2)
            sample_coords = RectifyCoordsGradient.apply(sample_coords, self.coords_lambda)
            _sample_coords = self.identity(sample_coords)

            k_selected = F.grid_sample(k, grid=sample_coords, padding_mode='zeros', align_corners=True)
            v_selected = F.grid_sample(v, grid=sample_coords, padding_mode='zeros', align_corners=True)

            q = rearrange(q, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
            # k = k_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
            k = rearrange(k_selected, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
            # v = v_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
            v = rearrange(v_selected, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
        else:
            transform_window_coords_distance = None
            q = rearrange(q, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
            # k = k_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
            k = rearrange(k, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
            # v = v_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
            v = rearrange(v, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rpe == 'v1':
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn += relative_position_bias.unsqueeze(0)
            pass
        elif self.rpe == 'v2':
            # q = rearrange(q, '(b hh ww) h (ws1 ws2) dim -> b h (hh ws1 ww ws2) dim', b=b, h=self.num_heads, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=self.window_size, ws2=self.window_size)
            # with torch.cuda.amp.autocast(enable=False):
            attn = calc_rel_pos_spatial(attn.float(), q.float(), (self.window_size, self.window_size), (self.window_size, self.window_size), self.rel_pos_h.float(), self.rel_pos_w.float())
        attn = attn.softmax(dim=-1)
        _attn = self.identity_attn(rearrange(attn, '(b hh ww) h ws1 ws2 -> (b h) (hh ww) ws1 ws2', b=b, h=self.num_heads//self.window_num, ww=window_num_w, hh=window_num_h, ws1=ws**2, ws2=ws**2))
        if transform_window_coords_distance is not None:
            transform_window_coords_distance = (transform_window_coords_distance * attn).sum(dim=-1)
            transform_window_coords_distance = self.identity_distance(transform_window_coords_distance)

        out = attn @ v
        out = rearrange(out, '(b hh ww) h (ws1 ws2) dim -> b (h dim) (hh ws1) (ww ws2)', h=self.num_heads//self.window_num, b=b, hh=window_num_h, ww=window_num_w, ws1=ws, ws2=ws)
        if padding_t + padding_d + padding_l + padding_r > 0:
            out = out[:, :, padding_t:h+padding_t, padding_l:w+padding_l]
        # globel_out.append(out)
        
        # globel_out = torch.stack(globel_out, dim=0)
        # out = rearrange(out, 'b c hh ww -> b (wsnum c) hh ww', wsnum=1, c=self.dim, b=b, hh=h, ww=w)
        out = out.reshape(b, self.dim, -1).permute(0, 2, 1)
        out = self.proj(out)
        return out

    def _reset_parameters(self):
        nn.init.constant_(self.transform[-1].weight, 0.)
        nn.init.constant_(self.transform[-1].bias, 0.)

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

class RectifyCoordsGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, coords_lambda=20):
        ctx.in1 = coords_lambda
        ctx.save_for_backward(coords)
        return coords

    @staticmethod
    def backward(ctx, grad_output):
        coords_lambda = ctx.in1
        coords, = ctx.saved_tensors
        grad_output[coords < -1.001] += -coords_lambda * 10
        grad_output[coords > 1.001] += coords_lambda * 10
        # print(f'coords shape: {coords.shape}')
        # print(f'grad_output shape: {grad_output.shape}')
        # print(f'grad sum for OOB locations: {grad_output[coords<-1.5].sum()}')
        # print(f'OOB location num: {(coords<-1.5).sum()}')

        return grad_output, None

def calc_rel_pos_spatial(
    attn,
    q,
    q_shape,
    k_shape,
    rel_pos_h,
    rel_pos_w,
    overlap=0
    ):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    k_h = k_h + 2 * overlap
    k_w = k_w + 2 * overlap

    # Scale up rel pos if shapes for q and k are different.
    # q_h_ratio = max(k_h / q_h, 1.0)
    # k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] - torch.arange(k_h)[None, :]
    )
    dist_h += (k_h - 1)
    # q_w_ratio = max(k_w / q_w, 1.0)
    # k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] - torch.arange(k_w)[None, :]
    )
    dist_w += (k_w - 1)

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn






if __name__ == '__main__': 
    model = QFormer(img_size=224,
                patch_size=4,  #4
                in_chans=3,
                num_classes=1000,
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[ 4, 8, 16, 32 ],     #     [ 4, 8, 16, 32 ]
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                drop_path_rate=0.1,          
                ape=False,
                patch_norm=True,
                use_checkpoint=False,
                coords_lambda=0.0,
                rpe='v1',
                )

# 尝试加载权重
    missing_keys, unexpected_keys = model.load_state_dict(torch.load('/data3/YG/FRINet/code/pre/QFormer_B_patch4_window7_224.pth'), strict=False)

    # 输出缺失和意外的键，以便调试
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
