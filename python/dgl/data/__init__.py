"""Data related package."""
from __future__ import absolute_import

from . import citation_graph as citegrh
from .citation_graph import CoraBinary, CitationGraphDataset, CoraDataset
from .minigc import *
from .tree import *
from .utils import *
from .sbm import SBMMixture
from .reddit import RedditDataset
from .ppi import PPIDataset, LegacyPPIDataset
from .tu import TUDataset, LegacyTUDataset
from .gnn_benckmark import AmazonCoBuy, CoraFull, Coauthor
from .karate import KarateClub
from .gindt import GINDataset
from .bitcoinotc import BitcoinOTC
from .gdelt import GDELT
from .icews18 import ICEWS18
from .qm7b import QM7b


def register_data_args(parser):
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help=
        "The input dataset. Can be cora, citeseer, pubmed, syn(synthetic dataset) or reddit"
    )
    citegrh.register_args(parser)


def load_data(args):
    if args.dataset == 'cora':
        return citegrh.load_cora()
    elif args.dataset == 'citeseer':
        return citegrh.load_citeseer()
    elif args.dataset == 'pubmed':
        return citegrh.load_pubmed()
    elif args.dataset == 'nell.0.001':
        return citegrh.load_nell_0_001()
    elif args.dataset == 'nell.0.01':
        return citegrh.load_nell_0_01()
    elif args.dataset == 'nell.0.1':
        return citegrh.load_nell_0_1()
    elif args.dataset == 'RMAT':
        return citegrh.load_RMAT('enwiki',100,10)
    elif args.dataset == 'syn':
        return citegrh.load_synthetic(args)
    elif args.dataset == 'CoraFull':
        return CoraFull()
    elif args.dataset == 'Coauthor_cs':
        return Coauthor('cs')
    elif args.dataset == 'Coauthor_physics':
        return Coauthor('physics')
    elif args.dataset == 'AmazonCoBuy_computers':
        return AmazonCoBuy('computers')
    elif args.dataset == 'AmazonCoBuy_photo':
        return AmazonCoBuy('photo')
    elif args.dataset is not None and args.dataset.startswith('reddit'):
        return RedditDataset(self_loop=('self-loop' in args.dataset))
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
