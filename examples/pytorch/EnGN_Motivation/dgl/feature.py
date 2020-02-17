import time
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
from model.gcn_gated import GCN_GATED
from model.graphsage import GraphSAGE
from dgl.nn.pytorch.conv import SGConv
from model.tagcn import TAGCN

from model.rgcn_ import RGCN_Class

root = './result/feature/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
#Amazon = citation_graph.load_RMAT('amazon',100,10)
#M3 = citation_graph.load_RMAT('3M',100,10)
#3M_276M = citation_graph.load_RMAT('3M_276M',100,10)
#_21M = citation_graph.load_RMAT('21',100,10)
#_22M = citation_graph.load_RMAT('22',50,10)
#_23M = citation_graph.load_RMAT('23',16,10)
#24M = citation_graph.load_RMAT('24',100,10)
#25M = citation_graph.load_RMAT('25',100,10)
#26M = citation_graph.load_RMAT('26',100,10)
F_64 = citation_graph.load_RMAT('feature',64,10)
F_128 = citation_graph.load_RMAT('feature',128,10)
F_256 = citation_graph.load_RMAT('feature',256,10)
F_512 = citation_graph.load_RMAT('feature',512,10)
F_1024 = citation_graph.load_RMAT('feature',1024,10)
F_2048 = citation_graph.load_RMAT('feature',2048,10)
F_4096 = citation_graph.load_RMAT('feature',4096,10)


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
DatasetList = [F_64, F_128, F_256, F_512, F_1024, F_2048, F_4096]
#DatasetList = [F_64]
ModelList   = [GCN]
#ModelList   = [GCN, GCN_GRU, GCN_GATED, GraphSAGE]
result = []
for d in DatasetList:
    g = DGLGraph(d.graph)
    norm = torch.pow(g.in_degrees().float(), -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1).to(device)
    for Net in ModelList: 
        time_now = int(time.time())
        time_local = time.localtime(time_now)
        day = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
        print('---start---{} - {} - {}'.format(d.name, Net.__name__, day))
        result.append('{},{},{}'.format(day, d.name, Net.__name__))
        model = Net(g, d.features.shape[1], d.num_labels).to(device)
        t,t_val = inference(model, d, epochs=500, device=device)
        time_now = int(time.time())
        time_local = time.localtime(time_now)
        day = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
        print('---end-----{} - {} - {}'.format(d.name, Net.__name__, day))
        result.append('{},{},{}'.format(day, d.name, Net.__name__))

path = root + 'result.csv'
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
