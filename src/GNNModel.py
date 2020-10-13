embed_dim = 3
import torch
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import numpy as np
from src.SAGEConv import SAGEConv
from torch_geometric.nn import GCNConv
from torch.nn import LogSoftmax


class GNNModel(torch.nn.Module):
    def __init__(self, num_teams):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(embed_dim, 16)
        self.conv2 = GCNConv(16, 3)

        # self.conv1 = SAGEConv(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        # self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings=num_teams, embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(6, 16)
        self.lin2 = torch.nn.Linear(16, 6)
        self.lin3 = torch.nn.Linear(6, 3)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()
        self.out = LogSoftmax(dim=0)

    def forward(self, data, home, away):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)
        x = x.squeeze(1)

        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))

        x = torch.cat([x[home],x[away]], dim=-1)




        # x, edge_index, _, batch, _,_ = self.pool1(x, edge_index, None, batch)
        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #
        # x = F.relu(self.conv2(x, edge_index))
        #
        # x, edge_index, _, batch, _,_ = self.pool2(x, edge_index, None, batch)
        # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #
        # x = F.relu(self.conv3(x, edge_index))
        #
        # x, edge_index, _, batch, _,_ = self.pool3(x, edge_index, None, batch)
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #
        # x = x1 + x2 + x3
        #
        # x = self.lin1(x)
        # x = self.act1(x)
        # x = self.lin2(x)
        # x = self.act2(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        #
        # # mad bullshit:
        # # ei_home = (edge_index[0] == home).nonzero().t()[0]
        # # ei_away = (edge_index[1] == away).nonzero().t()[0]
        # # node_indeces = np.intersection1(ei_home, ei_away)
        #
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = F.leaky_relu(self.lin3(x))
        # .squeeze(1)
        # x = self.out(x)
        return x.reshape(-1,3)