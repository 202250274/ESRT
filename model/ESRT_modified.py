import torch.nn as nn
import torch.nn.functional as F
from util.tools import extract_image_patches,\
    reduce_mean, reduce_sum, same_padding, reverse_patches

from util.transformer import drop_path, DropPath, PatchEmbed, Mlp,MLABlock
from util.position import PositionEmbeddingLearned, PositionEmbeddingSine
import os
import math
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, relu=True,
                 bn=False, bias=False, up_size=0,fan=False):
        super(BasicConv, self).__init__()
        wn = lambda x:torch.nn.utils.weight_norm(x)
        self.out_channels = out_planes
        self.in_channels = in_planes
        if fan:
            self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x





class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)





def default_conv(in_channels, out_channels, kernel_size, bias=True, groups = 1):
    wn = lambda x:torch.nn.utils.weight_norm(x)
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, groups = groups)




## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



scale = 2
n_feats = 8
n_channels = 1
kernel_size= 3
class SRCNN(nn.Module):
    def __init__(self,conv=default_conv):
        super(SRCNN, self).__init__()
        #self.conT1 = nn.ConvTranspose2d(n_feats, n_feats, kernel_size=2, stride= 2, padding=0,output_padding=0)
        self.upin = nn.Sequential(Upsampler(conv,scale,8*n_feats,act=False),
                          BasicConv(8*n_feats, 8*n_feats,3,1,1))
        
        self.conv1 = nn.Conv2d(n_feats, 2*n_feats, kernel_size=3, padding=1, padding_mode='replicate') # padding mode same as original Caffe code
        self.conv12 = nn.Conv2d(2*n_feats, 4*n_feats, kernel_size=3, padding=1, padding_mode='replicate') # padding mode same as original Caffe code
        self.conv13 = nn.Conv2d(4*n_feats, 8*n_feats, kernel_size=3, padding=1, padding_mode='replicate') # padding mode same as original Caffe code

        self.conv23 = nn.Conv2d(8*n_feats, 4*n_feats, kernel_size=3, padding=1, padding_mode='replicate')
        self.conv22 = nn.Conv2d(4*n_feats, 2*n_feats, kernel_size=3, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(2*n_feats, n_feats, kernel_size=3, padding=1, padding_mode='replicate')
        
        
        self.conv3 = nn.Conv2d(4*n_feats, 1, kernel_size=3, padding=1, padding_mode='replicate')
        
        self.up = nn.Sequential(Upsampler(conv,scale,n_feats,act=False),
                          BasicConv(n_feats, n_channels,3,1,1))
        
        modules_head = [conv(n_channels, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)
        
        self.atten1 = CALayer(2*n_feats)
        self.atten2 = CALayer(4*n_feats)
        self.atten3 = CALayer(8*n_feats)

        self.atten11 = CALayer(4*n_feats)
        self.atten22 = CALayer(2*n_feats)
        self.atten33 = CALayer(1*n_feats)
        
        self.down = nn.AvgPool2d(kernel_size=2)

        self.d1 = nn.Sequential(Upsampler(conv,scale,4*n_feats,act=False),
                          BasicConv(4*n_feats, 4*n_feats,3,1,1))
        
        self.d2 = nn.Sequential(Upsampler(conv,scale,2*n_feats,act=False),
                          BasicConv(2*n_feats, 2*n_feats,3,1,1))
        
        self.d3 = nn.Sequential(Upsampler(conv,scale,n_feats,act=False),
                          BasicConv(n_feats, n_feats,3,1,1))
        
        self.attention = MLABlock(n_feat=8*n_feats, dim=576) 
        self.alise = default_conv(8*n_feats, 8*n_feats, 3)
        
    def forward(self, x):
        xx = self.head(x)
        res2 = xx
        
        
        #x1 = F.relu(self.upin(xx))
        
        #Encoder
        x2 = F.relu(self.conv1(xx))
        x2 = self.atten1(x2)
        #x2 = self.down(x2)
        
        x2 = F.relu(self.conv12(x2))
        x2 = self.atten2(x2)
        #x2 = self.down(x2)
        
        x2 = F.relu(self.conv13(x2))
        x2 = self.atten3(x2)
        x2 = self.down(x2)
        
        
        # Transformer
        b,c,h,w = x2.shape
        #print(b,c,h,w )
        #out = self.attention(self.reduce(torch.cat([x1,x2,x3],dim=1)))
        x2 = self.attention(x2)
        x2 = x2.permute(0,2,1)
        x2 = reverse_patches(x2, (h,w), (3,3), 1, 1)
        x2 = self.alise(x2)
        x2 = F.relu(self.upin(x2))
        
        #Decoder
        x3 = F.relu(self.conv23(x2))
        x3 = self.atten11(x3)
        x3 = self.d1(x3)
        
        #x3 = F.relu(self.conv22(x3))
        #x3 = self.atten22(x3)
        #x3 = self.d2(x3)

        
        #x3 = F.relu(self.conv2(x3))
        #x3 = self.atten33(x3)
        #x3 = self.d3(x3)

        #Final Layer
        x4 = self.conv3(x3)

        #x4 = self.conv3(x2)
          
        #Up-Scale
        x11 =  self.up(res2)

        x4 = x4 + x11
        #print(x.size())
        return x4