import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from src.DataTransformer import DataTransformer


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
        # dt.read_data()

        # process by session_id
        grouped = dt.data.groupby(['league'])  # ,'year'
        for league_year, group in tqdm(grouped):
            # group = dt.clean_data(group)
            data_train, data_val, data_test, teams_enc = dt.prepare_data(data=group)
            n_teams = len(teams_enc['teams'].values)
            # win_lose_network = [{'won': set(), 'lost': set()} for _ in range(dt.n_teams)]
            win_lose_network = np.zeros((n_teams, 2, n_teams))

            # random_list = [x/1000 for x in random.sample(range(0, 1000), len(teams_enc['teams'].values))]
            # node_features = torch.FloatTensor(random_list).unsqueeze(1)
            # edge_index = torch.tensor(list(permutations(list(teams_enc['label_encoding']),2)), dtype=torch.long)

            # x = node_features
            # x = torch.tensor(teams_enc['label_encoding'].values).to(torch.int64)
            x = torch.ones(n_teams).reshape(-1, 1)
            # y = torch.FloatTensor([group.lwd.values])

            edge_time = np.empty((n_teams, n_teams))
            edge_time[:] = None

            node_time = np.zeros(n_teams)

            won = data_test[data_test['result'] == "W"].shape[0]
            lost = data_test[data_test['result'] == "L"].shape[0]
            draw = data_test[data_test['result'] == "D"].shape[0]

            data = Data(
                x=x,
                # edge_index=edge_index.t().contiguous(),
                matches=data_train,
                n_teams=n_teams,
                win_lose_network=win_lose_network,
                node_time=node_time,
                node_weight=None,
                edge_time=edge_time,
                data_val=data_val,
                data_test=data_test,
                curr_time=0,
                N=dt.N,
                baseline=max(won, lost, draw),
                running_loss=[],
                running_accuracy=[]
            )
            data_list.append(data)

        data, slices = self.collate(data_list)
        return data_list
        # self.matches = dt.data
        # torch.save((data, slices), self.processed_paths[0])
