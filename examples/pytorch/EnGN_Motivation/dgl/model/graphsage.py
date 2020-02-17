import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from model.kernel.sageconv import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_classes,
                 n_layers=1,
                 n_hidden=16,
                 activation=F.relu,
                 dropout=0.5,
                 aggregator_type='pool'):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None)) # activation None

    def forward(self, features):
        h = features
        stage1 = 0
        stage2 = 0
        stage3 = 0
        for layer in self.layers:
            h,s1,s2,s3 = layer(self.g, h)
            stage1 += s1
            stage2 += s2
            stage3 += s3
        return h, stage1, stage2, stage3
