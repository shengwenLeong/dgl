"""Torch modules for graph convolutions."""
# pylint: disable= no-member, arguments-differ, invalid-name

from .agnnconv import AGNNConv
from .appnpconv import APPNPConv
from .chebconv import ChebConv
from .edgeconv import EdgeConv
from .gatconv import GATConv
from .ginconv import GINConv
from .gmmconv import GMMConv
from .graphconv import GraphConv
from .gcn_gru import GraphConv_GRU
from .gcn_gated import GraphConv_Gated
from .nnconv import NNConv
from .relgraphconv import RelGraphConv
from .sageconv import SAGEConv
from .sgconv import SGConv
from .tagconv import TAGConv
from .gatedgraphconv import GatedGraphConv
from .densechebconv import DenseChebConv
from .densegraphconv import DenseGraphConv
from .densesageconv import DenseSAGEConv

__all__ = ['GraphConv', 'GraphConv_GRU', 'GraphConv_Gated', 'GATConv', 'TAGConv', 'RelGraphConv', 'SAGEConv',
           'SGConv', 'APPNPConv', 'GINConv', 'GatedGraphConv', 'GMMConv',
           'ChebConv', 'AGNNConv', 'NNConv', 'DenseGraphConv', 'DenseSAGEConv',
           'DenseChebConv', 'EdgeConv']
