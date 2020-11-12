import torch
from DataTransformer import DataTransformer
from Dataset import Dataset
from FlatModel import FlatModel
from Trainer import train_gnn_model, test_gnn_model, correct_by_class, evaluate, train_pr, train_flat_model, test_flat_model
from torch_geometric.data import DataLoader
from GNNModel import GNNModel
from PRDataset import PRDataset
from PageRank import PageRank
from utils import visualize_acc_loss
import pickle
# sacred
# from sacred import Experiment
# from sacred.observers import FileStorageObserver


def common_loading(filename):
    dt = DataTransformer(filename)
    # home, away, label = dt.get_train_data()
    print("Information on the whole dataset: ")
    dt.print_metadata(dt.data, "whole")
    return dt

def run_flat_model(filename):
    dt = common_loading(filename)
    # -----------Flat model---------------------
    model = FlatModel(dt.n_teams, 3)
    home, away, label = dt.get_train_data()
    train_flat_model(zip(home, away, label), model)
    PATH = './net.pth'
    torch.save(model.state_dict(), PATH)
    # net = Net()
    # net.load_state_dict(torch.load(PATH))
    # home, away, label = dt.get_test_data()
    # test_model(zip(home, away, label), model)
    # correct_by_class(zip(home, away, label), model)
    # pass


def run_pr_model(filename):
    # ----------PageRank------------------------
    dataset = PRDataset(root='../', filename=filename)
    dataset.process()
    batch_size = 1
    train_loader = DataLoader(dataset, batch_size=batch_size)
    model = PageRank()
    train_pr(train_loader, model)


def run_gnn_model(filename, lr=(0.001, 0.0001), exp_num=0, **kwargs):
    # ----------GNN------------------------------
    dataset = PRDataset(filename=filename)
    data_list = dataset.process()
    epochs = [500, 300]
    for i, data in enumerate(data_list):
        model = GNNModel(data.n_teams, **kwargs)
        train_gnn_model(data, model, epochs=epochs[0], lr=lr[0])
        train_gnn_model(data, model, epochs=epochs[1], dataset="val", lr=lr[1])
        test_acc = test_gnn_model(data, model, "test")
        file = outfile.format(pickle_dir, i, exp_num, "pickle")
        data_to_save = {"data": data, "model": model, "epochs": epochs}
        save_to_pickle(file, data_to_save)
        # evaluate(train_loader, model)
        visualize_acc_loss(data, epochs, outfile.format(images_dir, i, exp_num, "png"))
        return test_acc


def save_to_pickle(filename, data):
    with open(filename, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data


def grid_search(filename, outfile):
    embed_dim = [1, 2, 3, 5, 10]
    n_conv = [1, 2, 3]
    dims = [(1, 1, 1), (2, 2, 2), (2, 4, 2), (4, 4, 4)]
    lr = [(0.001, 0.0001), (0.0001, 0.001), (0.001, 0.001), (0.0001, 0.0001)]
    exp_counter = 1
    for e in embed_dim:
        for n in n_conv:
            for d in dims:
                for l in lr:
                    with open(outfile, "a+") as f:
                        acc = run_gnn_model(filename,l, exp_counter, embed_dim=e, n_conv=n, conv_dims=d)
                        f.write("EXP:[{}] embed_dim={}, n_conv={}, conv_dims={}, l={} achieved accuracy:{}\n".
                                format(exp_counter, e, n, d, l, acc))
                        print("EXP:[{}] embed_dim={}, n_conv={}, conv_dims={}, l={} achieved accuracy:{}\n".
                                format(exp_counter, e, n, d, l, acc))
                        exp_counter += 1


if __name__ == '__main__':
    model_id = 4  # 0:Flat, 1:PageRank, 2: GNN, 3: visualization, 4: grid search on gnn
    exp_num = "0"
    filename = "../data/GER1_2001.csv"
    # filename = "../data/0_test.csv"
    # filename = "../data/mini_data.csv"
    filename = "../data/GER1_all.csv"

    outfile = "{}data_{}_model_{}.{}"
    pickle_dir = "../data/models/"
    images_dir = "../data/img/"
    grid_search_file = "../data/grid_search_results.txt"

    if model_id == 0:
        run_flat_model(filename)
    elif model_id == 1:
        run_pr_model(filename)
    elif model_id == 2:
        run_gnn_model(filename)
    elif model_id == 4:
        grid_search(filename, grid_search_file)
    else:
        file = outfile.format(pickle_dir, 0,exp_num, "pickle")
        data = load_from_pickle(file)
        visualize_acc_loss(data["data"], data["epochs"], outfile.format(images_dir, 0, exp_num,"png"))
