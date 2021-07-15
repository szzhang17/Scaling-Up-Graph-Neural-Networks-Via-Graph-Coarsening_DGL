from dgl.nn import GraphConv, APPNPConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.conv2(g, h)
        return h

class APPNP(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, k, alpha):
        super(APPNP, self).__init__()
        self.lin1 = Linear(in_feats, h_feats)
        self.lin2 = Linear(h_feats, num_classes)
        self.propagate = APPNPConv(k, alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, g, in_feat):

        h = in_feat
        h = F.dropout(h, training=self.training)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, training=self.training)
        h = self.lin2(h)
        h = self.propagate(g, h)

        return h
