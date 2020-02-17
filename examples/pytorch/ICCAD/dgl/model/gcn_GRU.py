import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv_GRU

class GCN_GRU(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_classes,
                 n_hidden=16,
                 n_layers=1,
                 activation=F.relu,
                 dropout=0.5):
        super(GCN_GRU, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv_GRU(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv_GRU(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv_GRU(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
