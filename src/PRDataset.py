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
            # win_lose_network = [{'won': set(), 'lost': set()} for _ in range(dt.n_teams)]
            win_lose_network = np.zeros((dt.n_teams, 2, dt.n_teams))

            # random_list = [x/1000 for x in random.sample(range(0, 1000), len(teams_enc['teams'].values))]
            # node_features = torch.FloatTensor(random_list).unsqueeze(1)
            # edge_index = torch.tensor(list(permutations(list(teams_enc['label_encoding']),2)), dtype=torch.long)

            # x = node_features
            # x = torch.tensor(teams_enc['label_encoding'].values).to(torch.int64)
            x = torch.ones(dt.n_teams).reshape(-1, 1)
            # y = torch.FloatTensor([group.lwd.values])

            edge_time = np.empty((dt.n_teams, dt.n_teams))
            edge_time[:] = None

            node_time = np.zeros(dt.n_teams)

            data = Data(
                x=x,
                # edge_index=edge_index.t().contiguous(),
                matches = group,
                n_teams = dt.n_teams,
                win_lose_network = win_lose_network,
                node_time = node_time,
                node_weight = None,
                edge_time = edge_time,
                n_datapoints = group.shape[0]
                # , y=y
            )
            data_list.append(data)

        data, slices = self.collate(data_list)
        # self.matches = dt.data
        torch.save((data, slices), self.processed_paths[0])
