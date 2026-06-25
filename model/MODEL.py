import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.decomposition import PCA

from .GraphConv import GraphConvolution


class GCN_Vertex(nn.Module):
    """
    (N, C)
    """
    def __init__(self, c_in, c_hid, c_out, dropout):
        super(GCN_Vertex, self).__init__()
        self.gc1 = GraphConvolution(c_in, c_hid)
        self.gc2 = GraphConvolution(c_hid, 2 * c_hid)
        self.dropout = nn.Dropout2d(dropout)
        self.gc3 = GraphConvolution(2 * c_hid, c_out)

    def forward(self, x, adj):
        x = torch.tanh(self.gc1(x, adj))
        x = torch.tanh(self.gc2(x, adj))
        feat = x
        x = self.dropout(x)
        x = self.gc3(x, adj)

        return x, feat


class GCN_Edge(nn.Module):
    """
    (N, C)
    """
    def __init__(self, c_in, c_hid):
        super(GCN_Edge, self).__init__()
        self.gc1 = GraphConvolution(c_in, c_hid)
        self.gc2 = GraphConvolution(c_hid, 2 * c_hid)

    def forward(self, x, adj):
        x = torch.sigmoid(self.gc1(x, adj))
        x = torch.sigmoid(self.gc2(x, adj))
        return x


class conv_block(nn.Module):
    def __init__(self, c_in, c_out):
        super(conv_block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class downsample(nn.Module):
    def __init__(self, c_in):
        super(downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=3, stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(c_in),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class upsample(nn.Module):
    def __init__(self, c_in, c_out):
        super(upsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.layer(x)
        return x


class Mulit_layer_Encoder(nn.Module):
    """
    (H, W, 3) -> (H, W, 3)
    """
    def __init__(self):
        super(Mulit_layer_Encoder, self).__init__()

    def forward(self, x, feats):


class HLGSG(nn.Module):
    def __init__(self, c_img, c_m2f, c_out, num_heads=4, lambda_g=0.5, lambda_l=0.5):
        super(HLGSG, self).__init__()

    def cross_attention(self, texture_feat, semantic_feat):

    def forward(self, texture_feat, semantic_feat):

