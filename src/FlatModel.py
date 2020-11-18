import torch
from torch.nn import Embedding, Sequential as Seq, Linear, LeakyReLU, SELU, Softmax, Dropout, BatchNorm1d
import torch.nn.functional as F
from torch.nn import LogSoftmax


class FlatModel(torch.nn.Module):
    def __init__(self, n_teams, result, emb_dim=3,  pretrained_weights=None):
        super(FlatModel, self).__init__()
        # set hyperparameters for the model
        self.n_teams = n_teams
        self.out_dim = result
        # set the layers to be used in the model
        if pretrained_weights is not None:
            self.embedding = Embedding.from_pretrained(pretrained_weights)
        else:
            self.embedding = Embedding(n_teams, emb_dim)
        self.activation = LeakyReLU(0.2)

        self.hidden_0 = Linear(emb_dim*2, 64)
        self.hidden_1 = Linear(64, 32)
        self.hidden_2 = Linear(32, 16)
        self.hidden_3 = Linear(16, 6)
        self.hidden_4 = Linear(64, 6)
        self.out = Linear(6, result)
        self.out_act = LogSoftmax(dim=0)
        self.drop = Dropout(p=0.2)

    def forward(self, team_home, team_away):
        # should the embedding layer be flattened?
        home_emb = self.embedding(team_home)
        away_emb = self.embedding(team_away)
        cat = torch.cat((home_emb, away_emb), -1)
        # x = cat.view(1, -1)
        x = self.activation(self.hidden_0(cat))
        x = self.activation(self.hidden_1(x))
        # x = self.drop(x)
        x = self.activation(self.hidden_2(x))
        # x = self.drop(x)
        x = self.activation(self.hidden_3(x))
        # x = self.hidden_4(x)
        x = x.view(-1, 6)

        # team_home = team_home.view(-1,1)
        # team_away = team_away.view(-1,1)
        # cat = torch.cat((team_home, team_away), -1)
        # x = cat.view(-1, 6)
        x = self.out(x)
        x = self.out_act(x)
        return x
