import torch
from torch.nn import LogSoftmax, ReLU, Tanh, LeakyReLU
from torch_geometric.nn import GCNConv

target_dim = 3

activations = {
    'relu': ReLU(),
    'tanh': Tanh(),
    'leaky': LeakyReLU()
}


class GNNModel(torch.nn.Module):
    def __init__(self, num_teams, embed_dim=3, n_conv=3, conv_dims=(16, 16, 3, 16), n_dense=3, dense_dims=(8, 8, 8),
                 act_f='tanh', **kwargs):
        super(GNNModel, self).__init__()
        self.embed_dim = embed_dim
        self.n_conv = n_conv
        self.conv_dims = conv_dims
        self.n_dense = n_dense
        self.activation = activations[act_f]

        self.embedding = torch.nn.Embedding(num_embeddings=num_teams, embedding_dim=embed_dim)

        self.conv_layers = []

        self.conv_layers.append(GCNConv(self.embed_dim, self.conv_dims[0]))
        for i in range(n_conv - 1):
            self.conv_layers.append(GCNConv(conv_dims[i], conv_dims[i + 1]))

        self.lin_layers = []
        self.lin_layers.append(torch.nn.Linear(conv_dims[n_conv - 1] * 2, dense_dims[0]))
        for i in range(n_dense - 2):
            self.lin_layers.append(torch.nn.Linear(dense_dims[i], dense_dims[i + 1]))
        self.lin_layers.append(torch.nn.Linear(dense_dims[n_dense - 2], target_dim))

        self.out = LogSoftmax(dim=1)

    def forward(self, data, home, away):
        edge_index, edge_weight = data.edge_index, data.edge_weight
        # home, away = list(home), list(away)
        x = torch.tensor(list(range(data.n_teams)))
        x = self.embedding(x).reshape(-1, self.embed_dim)

        if len(edge_weight) > 0:
            x = self.conv_layers[0](x, edge_index, edge_weight )
        else:
            x = self.conv_layers[0](x, edge_index)
        x = self.activation(x)

        for i in range(self.n_conv - 1):
            if len(edge_weight) > 0:
                    x = self.activation(self.conv_layers[i + 1](x, data.edge_index, edge_weight))
            else:
                x = self.activation(self.conv_layers[i + 1](x, data.edge_index))

        x = torch.cat([x[home], x[away]], dim=1)

        for i in range(self.n_dense):
            x = self.activation(self.lin_layers[i](x))

        x = self.out(x)
        return x.reshape(-1, target_dim)
