import torch
from torch_geometric.nn import GCNConv
from torch import sigmoid as act, sign


class PageRank(torch.nn.Module):
    def __init__(self):
        super(PageRank, self).__init__()
        self.conv = GCNConv(1, 1, add_self_loops=False) #

    def forward(self, data, home, away, result):
        x = data.x.reshape(-1, 1)
        if result != 0:
            if result == 2:
                edge_index = torch.tensor([
                                            [home for i in range(len(data.win_lose_network[0][home]['lost']))]+[away],
                                           list(data.win_lose_network[0][home]['lost'])+[home]
                                           # [away],[home]
                                           ])
            elif result == 1:
                edge_index = torch.tensor([[away for i in range(len(data.win_lose_network[0][away]['lost']))]+[home],
                                            list(data.win_lose_network[0][away]['lost'])+[away]
                                           ])
            cache = [edge_index, torch.ones(len(edge_index[0]))]
            self.conv._cached_edge_index = cache
            self.conv.weight.data = torch.ones(1).view(1,1)
            x = self.conv(x, edge_index)
            data.x *= 0.8
            data.x += x
            data.x = act(data.x)
        output = data.x[home] - data.x[away]
        return sign(output)
