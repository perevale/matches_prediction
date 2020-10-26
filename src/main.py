import torch
from torch_geometric.data import Data
from src.DataTransformer import DataTransformer
from src.Dataset import Dataset
from src.FlatModel import FlatModel
from src.Trainer import train_model, test_model, correct_by_class, evaluate, train_pr
from torch_geometric.data import DataLoader
from src.GNNModel import GNNModel
from src.PRDataset import PRDataset
from src.PageRank import PageRank


# sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver

if __name__ == '__main__':
    filename = "../data/GER1_2001.csv"
    # filename = "../data/0_test.csv"
    filename = "../data/mini_data.csv"
    # filename = "../data/GER1_all.csv"
    dt = DataTransformer(filename)
    # home, away, label = dt.get_train_data()
    dt.print_metadata(dt.data)


    #----------PageRank------------------------
    dataset = PRDataset(root='../', filename=filename)
    dataset.process()
    batch_size = 1
    train_loader = DataLoader(dataset, batch_size=batch_size)
    model = PageRank()
    train_pr(train_loader, model )

    #----------GNN------------------------------
    # dataset = Dataset(root='../', filename=filename)
    # dataset.process()
    # batch_size = 1
    # train_loader = DataLoader(dataset, batch_size=batch_size)
    # model = GNNModel(dt.n_teams)
    # train_model(train_loader, model )
    # # evaluate(train_loader, model)

    #-----------Flat model---------------------
    # model = FlatModel(dt.n_teams, 3)
    # train_model(zip(home, away, label), model )
    # PATH = './net.pth'
    # torch.save(model.state_dict(), PATH)
    # # net = Net()
    # # net.load_state_dict(torch.load(PATH))
    # home, away, label = dt.get_test_data()
    # test_model(zip(home, away, label), model)
    # # correct_by_class(zip(home, away, label), model)
    # pass
