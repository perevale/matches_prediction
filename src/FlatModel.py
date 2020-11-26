import torch
from torch.nn import Embedding, Linear, Dropout
from torch.nn import LogSoftmax, ReLU, Tanh, LeakyReLU

activations = {
    'relu': ReLU(),
    'tanh': Tanh(),
    'leaky': LeakyReLU(0.2)
}


class FlatModel(torch.nn.Module):
    def __init__(self, n_teams, out_dim=3, embed_dim=3, pretrained_weights=None, n_dense=5, dense_dims=(64, 32, 16, 6),
                 act_f='leaky', **kwargs):
        super(FlatModel, self).__init__()
        # set hyperparameters for the model
        self.n_teams = n_teams
        self.out_dim = out_dim
        self.activation = activations[act_f]
        self.n_dense = n_dense

        # set the layers to be used in the model
        if pretrained_weights is not None:
            self.embedding = Embedding.from_pretrained(pretrained_weights)
        else:
            self.embedding = Embedding(n_teams, embed_dim)

        self.lin_layers = []
        self.lin_layers.append(torch.nn.Linear(embed_dim * 2, dense_dims[0]))
        for i in range(n_dense - 2):
            self.lin_layers.append(torch.nn.Linear(dense_dims[i], dense_dims[i + 1]))
        self.lin_layers.append(torch.nn.Linear(dense_dims[n_dense - 2], self.out_dim))

        self.out = LogSoftmax(dim=1)

        self.drop = Dropout(p=0.2)

    def forward(self, team_home, team_away):
        home_emb = self.embedding(team_home)
        away_emb = self.embedding(team_away)
        x = torch.cat((home_emb, away_emb), -1)

        for i in range(self.n_dense):
            x = self.activation(self.lin_layers[i](x))

        x = self.out(x)
        return x.reshape(-1, self.out_dim)
