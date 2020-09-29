import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from src.DataTransformer import DataTransformer
from itertools import permutations
import random


class Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

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

        filename = "../data/GER1_2001.csv"
        dt = DataTransformer(filename)
        dt.read_data()

        # process by session_id
        grouped = dt.data.groupby(['league','year'])
        for league_year, group in tqdm(grouped):
            # sess_item_id = LabelEncoder().fit_transform(group.item_id)

            group, teams_enc = dt.prepare_data(group)
            # group['sess_item_id'] = sess_item_id
            # group = group.reset_index(drop=True)
            # node_features = group.loc[group.league == league_year[0] and group.year == league_year[1], ['home_team','away_team', 'item_id']].item_id.drop_duplicates().values
            # len(teams_enc['teams'].values)
            # node_features = torch.LongTensor(node_features).unsqueeze(1)
            # target_nodes = group.sess_item_id.values[1:]
            # source_nodes = group.sess_item_id.values[:-1]
            #
            # edge_index = torch.tensor([source_nodes,
            #                            target_nodes], dtype=torch.long)
            node_features = torch.LongTensor(random.sample(range(0, 1), len(teams_enc['teams'].values))).unsqueeze(1)
            edge_index = torch.tensor(list(permutations(list(teams_enc['label_encoding']))), dtype=torch.long)

            x = node_features

            # y = torch.FloatTensor([group.label.values[0]])

            data = Data(x=x, edge_index=edge_index.t().contiguous())
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
