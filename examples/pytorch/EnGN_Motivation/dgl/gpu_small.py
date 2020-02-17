from itertools import product
import numpy as np
import torch
import dgl
from dgl.data import citation_graph, CoraFull, RedditDataset
from dgl.data import AmazonCoBuy, Coauthor
from dgl.contrib.data import load_data
from dgl import DGLGraph

from model.inference_time import inference
from model.gcn import GCN 
from model.gcn_GRU import GCN_GRU
from model.graphsage import GraphSAGE
#from model.graphsage_cpu import GraphSAGE
from model.gcn_gated import GCN_GATED
from dgl.nn.pytorch.conv import SGConv
from model.tagcn import TAGCN

from model.rgcn_ import RGCN_Class

root = './result/motivation/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Cora = citation_graph.load_cora()
CiteSeer = citation_graph.load_citeseer()
PubMed = citation_graph.load_pubmed()
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
g = DGLGraph(Cora.graph)
norm = torch.pow(g.in_degrees().float(), -0.5)
norm[torch.isinf(norm)] = 0
g.ndata['norm'] = norm.unsqueeze(1).to(device)
model = GCN(g, Cora.features.shape[1], Cora.num_labels).to(device)
inference(model, Cora, epochs=10, device=device)
print('---------start------------')
DatasetList = [Cora, CiteSeer, PubMed]
#DatasetList = [Cora]
ModelList   = [GCN, GraphSAGE]
result = []
for d in DatasetList:
    g = DGLGraph(d.graph)
    norm = torch.pow(g.in_degrees().float(), -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1).to(device)
    stage1 = '{},'.format(d.name)+'s1,'
    stage2 = '{},'.format(d.name)+'s2,'
    stage3 = '{},'.format(d.name)+'s3,'
    total  = '{},'.format(d.name)+'total,'
    for Net in ModelList: 
        model = Net(g, d.features.shape[1], d.num_labels).to(device)
        t_val, s1, s2, s3 = inference(model, d, epochs=20, device=device)
        print('{} - {}'.format(d.name, Net.__name__))
        stage1 += '{:3f},'.format(np.mean(s1))
        stage2 += '{:3f},'.format(np.mean(s2))
        stage3 += '{:3f},'.format(np.mean(s3))
        total  += '{:3f},'.format(np.mean(t_val))
    result.append(stage1+'\t\n'+stage2+'\t\n'+stage3+'\t\n'+total+'\t\n')

path = root + 'gpu_small_result.csv'
with open(path, 'w') as output_file:
    out_string = 'Dataset, ,'
    for val_ in ModelList:
        out_string += str(val_.__name__)+','
    output_file.write(out_string+'\t\n')
    for val_ in result:
        output_file.write(str(val_))
'''
for d, Net in product([AIFB, MUTAG, BGS, AM], [RGCN_Class]):
    g = DGLGraph()
    g.add_nodes(d.num_nodes)
    g.add_edges(d.edge_src, d.edge_dst)
    feats = (torch.arange(d.num_nodes)).to(device)
    edge_type = (torch.from_numpy(d.edge_type)).to(device)
    edge_norm = (torch.from_numpy(d.edge_norm).unsqueeze(1)).to(device)
    g.edata.update({'type': edge_type, 'norm': edge_norm})
    model = Net(g, d.num_classes, d.num_rels, use_cuda=1).to(device)
    t = inference(model, feats, epochs=10, device=device)
    print('{} - {}: {:.4f}s'.format(d.name, Net.__name__, t))
'''
