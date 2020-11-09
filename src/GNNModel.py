embed_dim = 3
target_dim = 3
import torch
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import numpy as np
from src.SAGEConv import SAGEConv
from torch_geometric.nn import GCNConv, ClusterGCNConv
from torch.nn import LogSoftmax

from src.utils import calculate_edge_weight, get_neighbour_edge_index


class GNNModel(torch.nn.Module):
    def __init__(self, num_teams):
        super(GNNModel, self).__init__()
        self.conv1 = ClusterGCNConv(embed_dim, 3)
        self.conv1 = GCNConv(embed_dim, 2)
        self.conv2 = GCNConv(2, 2)

        # self.conv1 = SAGEConv(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        # self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings=num_teams, embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(4, 6)
        self.lin2 = torch.nn.Linear(16, 6)
        self.lin3 = torch.nn.Linear(6, target_dim)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()
        self.out = LogSoftmax(dim=0)

    def forward(self, data, home, away):
        edge_index, edge_weight = data.edge_index, data.edge_weight
        x = torch.tensor(list(range(data.n_teams)))
        x = self.item_embedding(x).reshape(-1,embed_dim)
        # x = x.squeeze(1)

        # x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.conv1(x, edge_index, edge_weight)
        x = F.leaky_relu(x)


        data.edge_index = get_neighbour_edge_index(data)
        if len(data.edge_index) > 0:
            edge_weight = calculate_edge_weight(data)
            x = F.leaky_relu(self.conv2(x, data.edge_index, edge_weight))




        x = torch.cat([x[home],x[away]], dim=-1)

        x = F.leaky_relu(self.lin1(x))
        # x = F.leaky_relu(self.lin2(x))
        x = F.leaky_relu(self.lin3(x))
        # .squeeze(1)
        x = self.out(x)
        # x = F.softmax(x, dim=0)
        return x.reshape(-1, target_dim)