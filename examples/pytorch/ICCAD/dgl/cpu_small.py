
from itertools import product
import numpy as np
import torch
import dgl
from dgl.data import citation_graph, CoraFull, RedditDataset
from dgl.data import AmazonCoBuy, Coauthor
from dgl.contrib.data import load_data
from dgl import DGLGraph

from model.inference import inference
from model.gcn import GCN 
from model.gcn_GRU import GCN_GRU
from model.gcn_gated import GCN_GATED
#from model.graphsage import GraphSAGE
from model.graphsage_cpu import GraphSAGE
from dgl.nn.pytorch.conv import SGConv
from model.tagcn import TAGCN

from model.rgcn_ import RGCN_Class

root = './result/cpu_result/small/'

device = torch.device('cpu')
Cora = citation_graph.load_cora()
#CiteSeer = citation_graph.load_citeseer()
#PubMed = citation_graph.load_pubmed()
#Nell   = citation_graph.load_nell_0_001()
#Coauthor_cs = Coauthor('cs')
#Coauthor_physics = Coauthor('cs')
#Amazon_computer = AmazonCoBuy('computers')
#Amazon_photo = AmazonCoBuy('photo')

#CoraFull = CoraFull()
#Reddit = RedditDataset(self_loop=True)
#Enwiki = citation_graph.load_RMAT('enwiki',100,10)

#AIFB  = load_data('aifb', bfs_level=3)
#MUTAG = load_data('mutag', bfs_level=3)
#BGS   = load_data('bgs', bfs_level=3)
#AM    = load_data('am', bfs_level=3)

#One training run before we start tracking duration to warm up GPU.
print('---------start------------')
g = DGLGraph(Cora.graph)
norm = torch.pow(g.in_degrees().float(), -0.5)
norm[torch.isinf(norm)] = 0
g.ndata['norm'] = norm.unsqueeze(1).to(device)
model = GCN(g, Cora.features.shape[1], Cora.num_labels).to(device)
inference(model, Cora, epochs=1, device=device)
print('---------end------------')
