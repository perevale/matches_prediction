import torch
from torch_geometric.nn import GCNConv
from torch import sigmoid as act, sign


class PageRank(torch.nn.Module):
    def __init__(self):
        super(PageRank, self).__init__()
        self.conv = GCNConv(1, 1, add_self_loops=False) #

    def forward(self, data, home, away, training=True):
        x, edge_index = data.x.reshape(-1, 1), data.edge_index
        if training:
            edge_weight = torch.ones(len(edge_index[0]))
            cache = [edge_index, edge_weight]
            self.conv._cached_edge_index = cache
            self.conv.weight.data = torch.ones(1).view(1,1)
            x = self.conv(x, edge_index)
            data.x *= data.node_weight
            data.x += x
            data.x = act(data.x)
        output = data.x[home] - data.x[away]
        return sign(output)
