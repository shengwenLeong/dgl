import torch.nn as nn

class BaseRGCN(nn.Module):
    def __init__(self,
                 g,
                 out_dim, 
                 num_rels, 
                 h_dim=16,
                 num_bases=-1,
                 num_hidden_layers=0, 
                 dropout=0,
                 use_self_loop=False, 
                 use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.graph = g
        self.num_nodes = len(g) 
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, h):
        stage1 = 0
        stage2 = 0
        stage3 = 0
        for layer in self.layers:
            h, s1, s2, s3 = layer(self.graph, h, self.graph.edata['type'], self.graph.edata['norm'])
            stage1 += s1
            stage2 += s2
            stage3 += s3
        return h, stage1, stage2, stage3
