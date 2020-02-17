import torch
import torch.nn as nn
import torch.nn.functional as F
from model.kernel.graphconv import GraphConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_classes,
                 n_hidden=16,
                 n_layers=1,
                 activation=F.relu,
                 dropout=0.5):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        stage1 = 0
        stage2 = 0
        stage3 = 0
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h,s1,s2,s3, = layer(self.g, h)
            stage1 += s1
            stage2 += s2
            stage3 += s3
        return h, stage1, stage2, stage3
