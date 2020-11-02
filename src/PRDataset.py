import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from src.DataTransformer import DataTransformer
from itertools import permutations
import random
import numpy as np

class PRDataset(InMemoryDataset):
    def __init__(self, root, filename, transform=None, pre_transform=None):
        super(PRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.filename = filename

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['../data/GER.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []

        dt = DataTransformer(self.filename)
        dt.read_data()

        # process by session_id
        grouped = dt.data.groupby(['league'])#,'year'
        for league_year, group in tqdm(grouped):
            group = dt.clean_data(group)
            group, teams_enc = dt.prepare_data(data=group, split_to_test=False)
            win_lose_network = [{'won': set(), 'lost': set()} for _ in range(dt.n_teams)]

            # random_list = [x/1000 for x in random.sample(range(0, 1000), len(teams_enc['teams'].values))]
            # node_features = torch.FloatTensor(random_list).unsqueeze(1)
            # edge_index = torch.tensor(list(permutations(list(teams_enc['label_encoding']),2)), dtype=torch.long)

            # x = node_features
            # x = torch.tensor(teams_enc['label_encoding'].values).to(torch.int64)
            x = torch.ones(dt.n_teams).reshape(-1, 1)
            # y = torch.FloatTensor([group.lwd.values])

            # node_time = np.empty((dt.n_teams, dt.n_teams))
            # node_time[:] = None
            node_time = np.zeros(dt.n_teams)

            data = Data(
                x=x,
                # edge_index=edge_index.t().contiguous(),
                matches = group,
                n_teams = dt.n_teams,
                win_lose_network = win_lose_network,
                node_time = node_time,
                node_weight = None
                # , y=y
            )
            data_list.append(data)

        data, slices = self.collate(data_list)
        # self.matches = dt.data
        torch.save((data, slices), self.processed_paths[0])


def calculate_win_lose_network(group, n_teams):
    win_draw = group[(group.lwd == 2)] # (group.lwd == 1) |
    lose = group[(group.lwd == 0)]
    win_lose_network = [{'won': set(), 'lost': set()} for _ in range(n_teams)]
    for i in range(n_teams):
        # won
        home_team = win_draw[win_draw.home_team == i]
        win_lose_network[i]['won'].update(set(home_team.away_team))
        away_team = lose[lose.away_team == i]
        win_lose_network[i]['won'].update(set(away_team.home_team))
        # lost
        home_team = lose[lose.home_team == i]
        win_lose_network[i]['lost'].update(set(home_team.away_team))
        away_team = win_draw[win_draw.away_team == i]
        win_lose_network[i]['lost'].update(set(away_team.home_team))
    return win_lose_network


def update_win_lose_network(win_lose_network, record):
    if record.lwd == 2: # record.lwd == 1 or
        win_lose_network[record.home_team]['won'].add(record.away_team)
        win_lose_network[record.away_team]['lost'].add(record.home_team)
    elif record.lwd == 0:
        win_lose_network[record.home_team]['lost'].add(record.away_team)
        win_lose_network[record.away_team]['won'].add(record.home_team)
    else:
        pass
    # return win_lose_network


def create_edge_index(data, home, away, result):
    if result == 2:
        data.edge_index = torch.tensor([
            [home for i in range(len(data.win_lose_network[0][home]['lost']))] + [away],
            list(data.win_lose_network[0][home]['lost']) + [home]
        ])
    elif result == 1:
        data.edge_index = torch.tensor([[away for i in range(len(data.win_lose_network[0][away]['lost']))] + [home],
                                   list(data.win_lose_network[0][away]['lost']) + [away]
                                   ])


def update_node_time(data, curr_time):
    indeces = data.edge_index[:][1].numpy()
    data.node_time[0][indeces] = curr_time


def calculate_node_weight(data, curr_time, N):
    # if data.edge_time[home,away] is not None and data.edge_time[away] is not None:

    data.node_weight = torch.tensor(1 - ((curr_time - data.node_time[0]) / N).astype(np.float32)).reshape(-1,1)
