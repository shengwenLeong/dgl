"""Torch modules for graph convolutions(GCN)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
import torch
from torch import nn
from torch.nn import init

import dgl.function as fn

# pylint: disable=W0235
class GraphConv_GRU(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm=True,
                 bias=True,
                 activation=None):
        super(GraphConv_GRU, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        self.rnn = torch.nn.GRUCell(in_feats, out_feats, bias=bias)
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.weight)
        self.rnn.reset_parameters()
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat):
        graph = graph.local_var()
        if self._norm:
            norm = th.pow(graph.in_degrees().float().clamp(min=1), -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp).to(feat.device)
            feat = feat * norm

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            t_s1 = time.perf_counter()
            feat_ = th.matmul(feat, self.weight)
            t_s2 = time.perf_counter()
            graph.ndata['h'] = feat_
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
            t_s3 = time.perf_counter()
        else:
            # aggregate first then mult W
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
            rst = th.matmul(rst, self.weight)
        if self._norm:
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)
        rst_ = self.rnn(feat, rst)
        t_s4 = time.perf_counter()
        s1 = (t_s2 - t_s1) * 1000000
        s2 = (t_s3 - t_s2) * 1000000
        s3 = (t_s4 - t_s3) * 1000000
        return rst_, s1, s2, s3

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)
