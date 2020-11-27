from DataTransformer import DataTransformer
from Dataset import Dataset
from FlatModel import FlatModel
from Trainer import train_gnn_model, test_gnn_model, correct_by_class, evaluate, train_pr, train_flat_model, \
    test_flat_model, continuous_evaluation, test_cont_gnn, train_cont_gnn
from GNNModel import GNNModel
from PRDataset import PRDataset
from PageRank import PageRank
from LSTMModel import create_LSTM_model, cont_eval_LSTM, test_LSTM_model
from utils import visualize_acc_loss, save_to_pickle, load_from_pickle, save_keras_model

dir_prefix = "../"
outfile = "{}data_{}_model_{}.{}"
pickle_dir = "{}data/models/"
images_dir = "{}data/img/"


def common_loading(filename):
    dt = DataTransformer(filename)
    # home, away, label = dt.get_train_data()
    print("Information on the whole dataset: ")
    dt.print_metadata(dt.data, "whole")
    return dt


def run_flat_model(filename, dir_prefix="../", lr=(0.001, 0.0001), exp_num=0, **kwargs):
    dataset = PRDataset(filename=filename)
    data_list = dataset.process()
    epochs = [2000, 500]
    for i, data in enumerate(data_list):
        model = FlatModel(data.n_teams, **kwargs)
        train_flat_model(data, model, epochs=epochs[0], lr=lr[0], print_info=True)
        train_flat_model(data, model, epochs=epochs[1], dataset="val", lr=lr[1], print_info=True)
        test_flat_model(data, model, "test")
        print("accuracy on testing data is: {}".format(data.test_accuracy))
        file = outfile.format(pickle_dir.format(dir_prefix), i, exp_num, "pickle")
        data_to_save = {"data": data, "model": model, "epochs": epochs}
        save_to_pickle(file, data_to_save)
        visualize_acc_loss(data, epochs, outfile.format(images_dir.format(dir_prefix), i, exp_num, "png"))
        return data.test_accuracy

def run_LSTM_model(filename, dir_prefix="../", lr=(0.00001, 0.0001), exp_num=0, **kwargs):
    dataset = PRDataset(filename=filename)
    data_list = dataset.process()
    epochs = [30]
    for i, data in enumerate(data_list):
        model = create_LSTM_model(data.n_teams)
        cont_eval_LSTM(data, model)
        file = outfile.format(pickle_dir.format(dir_prefix), i, "LSTM_2", {})
        save_keras_model(file, model)
        data_to_save = {"data": data}
        file = file.format('pickle')
        save_to_pickle(file, data_to_save)


def run_pr_model(filename):
    # ----------PageRank------------------------
    dataset = PRDataset(filename=filename)
    data_list = dataset.process()
    for i, data in enumerate(data_list):
        model = PageRank()
        train_pr(data, model)


def run_gnn_cont(filename, dir_prefix="../", lr=(0.00001, 0.0001), exp_num=0, **kwargs):
    # ----------GNN------------------------------
    dataset = PRDataset(filename=filename)
    data_list = dataset.process()
    epochs = [30]
    for i, data in enumerate(data_list):
        model = GNNModel(data.n_teams, **kwargs)
        continuous_evaluation(data, model, epochs[0], batch_size=9)
        test_cont_gnn(data, model, data.data_test, "test")
        print("accuracy on testing data is: {}".format(data.test_accuracy))
        file = outfile.format(pickle_dir.format(dir_prefix), i, exp_num, "pickle")
        data_to_save = {"data": data, "model": model, "epochs": epochs}
        save_to_pickle(file, data_to_save)
        return data.test_accuracy


def run_gnn_model(filename, dir_prefix="../", lr=(0.00001, 0.0001), exp_num=0, **kwargs):
    # ----------GNN------------------------------
    dataset = PRDataset(filename=filename)
    data_list = dataset.process()
    epochs = [500, 300]
    for i, data in enumerate(data_list):
        model = GNNModel(data.n_teams, **kwargs)
        train_gnn_model(data, model, epochs=epochs[0], lr=lr[0])
        train_gnn_model(data, model, epochs=epochs[1], dataset="val", lr=lr[1])
        test_acc = test_gnn_model(data, model, "test")
        file = outfile.format(pickle_dir.format(dir_prefix), i, exp_num, "pickle")
        data_to_save = {"data": data, "model": model, "epochs": epochs}
        save_to_pickle(file, data_to_save)
        # evaluate(train_loader, model)
        visualize_acc_loss(data, epochs, outfile.format(images_dir.format(dir_prefix), i, exp_num, "png"))
        return test_acc


def run_gnn_batched(filename):
    dataset = PRDataset(filename=filename)
    data_list = dataset.process()
    for i, data in enumerate(data_list):
        model = GNNModel(data.n_teams)
        matches = data.matches.append(data.data_val, ignore_index=True)
        train_cont_gnn(data, matches, model, epochs=2000, lr=0.01, print_info=True)

def grid_search(filename, outfile):
    embed_dim = [1, 2, 3, 5, 10]
    n_conv = [1, 2, 3]
    dims = [(1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8)]
    lr = [(0.001, 0.0001), (0.0001, 0.001), (0.001, 0.001), (0.0001, 0.0001)]
    exp_counter = 0
    for e in embed_dim:
        for n in n_conv:
            for d in dims:
                for l in lr:
                    exp_counter += 1
                    if exp_counter < 25:
                        continue
                    with open(outfile, "a+") as f:
                        acc = run_gnn_model(filename, l, exp_counter, embed_dim=e, n_conv=n, conv_dims=d)
                        f.write("EXP:[{}] embed_dim={}, n_conv={}, conv_dims={}, l={} achieved accuracy:{}\n".
                                format(exp_counter, e, n, d, l, acc))
                        print("EXP:[{}] embed_dim={}, n_conv={}, conv_dims={}, l={} achieved accuracy:{}\n".
                              format(exp_counter, e, n, d, l, acc))


if __name__ == '__main__':
    # 0:Flat, 1:PageRank, 2: GNN, 3: visualization, 4: grid search on gnn, 5: gnn cont, 6: LSTM, 7: gnn batched
    model_id = 0
    exp_num = "0"
    filename = "../data/GER1_2001.csv"
    # filename = "../data/0_test.csv"
    # filename = "../data/mini_data.csv"
    filename = "../data/GER1_all.csv"

    # outfile = "{}data_{}_model_{}.{}"
    # pickle_dir = "../data/models/"
    # images_dir = "../data/img/"
    grid_search_file = "../data/grid_search_results.txt"

    if model_id == 0:
        run_flat_model(filename)
    elif model_id == 1:
        run_pr_model(filename)
    elif model_id == 2:
        run_gnn_model(filename)
    elif model_id == 4:
        grid_search(filename, grid_search_file)
    elif model_id == 5:
        run_gnn_cont(filename)
    elif model_id == 6:
        run_LSTM_model(filename)
    elif model_id == 7:
        run_gnn_batched(filename)
    else:
        file = outfile.format(pickle_dir.format(dir_prefix), 0, exp_num, "pickle")
        data = load_from_pickle(file)
        visualize_acc_loss(data["data"], data["epochs"],
                           outfile.format(images_dir.format(dir_prefix), 0, exp_num, "png"))
