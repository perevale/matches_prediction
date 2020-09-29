import torch
from torch_geometric.data import Data
from src.DataTransformer import DataTransformer
from src.Dataset import Dataset
from src.NetModel import NetModel
from src.Trainer import train_model, test_model, correct_by_class

# sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver

if __name__ == '__main__':
    filename = "../data/GER1_all.csv"

    # dataset = Dataset(root='../')
    # dataset.process()
    dt = DataTransformer(filename)
    home, away, label = dt.get_train_data()
    model = NetModel(dt.n_teams, 3)
    train_model(zip(home, away, label), model )
    PATH = './net.pth'
    torch.save(model.state_dict(), PATH)
    # net = Net()
    # net.load_state_dict(torch.load(PATH))
    home, away, label = dt.get_test_data()
    test_model(zip(home, away, label), model)
    # correct_by_class(zip(home, away, label), model)
    pass
