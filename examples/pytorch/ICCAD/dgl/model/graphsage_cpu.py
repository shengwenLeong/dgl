import argparse
import time
import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import dgl.function as fn


class Aggregator(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation=None, bias=True):
        super(Aggregator, self).__init__()
        self.g = g
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)  # (F, EF) or (2F, EF)
        self.activation = activation
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, node):
        nei = node.mailbox['m']  # (B, N, F)
        h = node.data['h']  # (B, F)
        h = self.concat(h, nei, node)  # (B, F) or (B, 2F)
        h = self.linear(h)   # (B, EF)
        if self.activation:
            h = self.activation(h)
        norm = torch.pow(h, 2)
        norm = torch.sum(norm, 1, keepdim=True)
        norm = torch.pow(norm, -0.5)
        norm[torch.isinf(norm)] = 0
        # h = h * norm
        return {'h': h}

    @abc.abstractmethod
    def concat(self, h, nei, nodes):
        raise NotImplementedError


class MeanAggregator(Aggregator):
    def __init__(self, g, in_feats, out_feats, activation, bias):
        super(MeanAggregator, self).__init__(g, in_feats, out_feats, activation, bias)

    def concat(self, h, nei, nodes):
        degs = self.g.in_degrees(nodes.nodes()).float()
        if h.is_cuda:
            degs = degs.cuda(h.device)
        #print(nei.shape)
        #print(h.shape)
        concatenate = torch.cat((nei, h.unsqueeze(1)), 1)
        #print(concatenate.shape)
        concatenate = torch.sum(concatenate, 1) / degs.unsqueeze(1)
        #print(concatenate.shape)
        return concatenate  # (B, F)


class PoolingAggregator(Aggregator):
    def __init__(self, g, in_feats, out_feats, activation, bias):  # (2F, F)
        super(PoolingAggregator, self).__init__(g, in_feats*2, out_feats, activation, bias)
        self.mlp = PoolingAggregator.MLP(in_feats, in_feats, F.relu, False, True)

    def concat(self, h, nei, nodes):
        nei = self.mlp(nei)  # (B, F)
        #print(nei.shape)
        #print(h.shape)
        concatenate = torch.cat((nei, h), 1)  # (B, 2F)
        #print(concatenate.shape)
        return concatenate

    class MLP(nn.Module):
        def __init__(self, in_feats, out_feats, activation, dropout, bias):  # (F, F)
            super(PoolingAggregator.MLP, self).__init__()
            self.linear = nn.Linear(in_feats, out_feats, bias=bias)  # (F, F)
            self.dropout = nn.Dropout(p=dropout)
            self.activation = activation
            nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

        def forward(self, nei):
            nei = self.dropout(nei)  # (B, N, F)
            nei = self.linear(nei)
            if self.activation:
                nei = self.activation(nei)
            #print(nei.shape)
            max_value = torch.max(nei, dim=1)[0]  # (B, F)
            #print(max_value.shape)
            return max_value


class GraphSAGELayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 aggregator_type,
                 bias=True,
                 ):
        super(GraphSAGELayer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(p=dropout)
        if aggregator_type == "pooling":
            self.aggregator = PoolingAggregator(g, in_feats, out_feats, activation, bias)
        else:
            self.aggregator = MeanAggregator(g, in_feats, out_feats, activation, bias)

    def forward(self, h):
        h = self.dropout(h)
        self.g.ndata['h'] = h
        self.g.update_all(fn.copy_src(src='h', out='m'), self.aggregator)
        h = self.g.ndata.pop('h')
        return h


class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_classes,
                 n_layers=1,
                 activation=F.relu,
                 n_hidden=16,
                 dropout=0.5,
                 aggregator_type='pooling'):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GraphSAGELayer(g, in_feats, n_hidden, activation, dropout, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphSAGELayer(g, n_hidden, n_hidden, activation, dropout, aggregator_type))
        # output layer
        self.layers.append(GraphSAGELayer(g, n_hidden, n_classes, None, dropout, aggregator_type))

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h
