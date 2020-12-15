from DataTransformer import DataTransformer
from Dataset import Dataset
from FlatModel import FlatModel
from Trainer import *
from GNNModel import GNNModel
from PRDataset import PRDataset
from PageRank import PageRank
from LSTMModel import create_LSTM_model, cont_eval_LSTM, test_LSTM_model
from utils import save_to_pickle, load_from_pickle, save_keras_model
from visualization import visualize_acc_loss, visualize_cont_eval, visualize_embedding

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


def run_cut_flat_model(filename, dir_prefix="../", lr=(0.001, 0.001), exp_num=0, **kwargs):
    dataset = PRDataset(filename=filename)
    data_list = dataset.process()
    epochs = [500, 500]
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


def run_flat_cont(filename, dir_prefix="../", lr=0.001, exp_num=0, **kwargs):
    # ----------Flat------------------------------
    dataset = PRDataset(filename=filename)
    data_list = dataset.process()
    epochs = [30]
    for i, data in enumerate(data_list):
        model = FlatModel(data.n_teams, **kwargs)
        print("Flat model")
        continuous_evaluation(data, model, epochs[0],lr=lr, batch_size=9)
        test_cont(data, model, data.data_test, "test")
        print("accuracy on testing data is: {}".format(data.test_accuracy))
        file = outfile.format(pickle_dir.format(dir_prefix), i, exp_num, "pickle")
        data_to_save = {"data": data, "model": model, "epochs": epochs}
        save_to_pickle(file, data_to_save)
        return data.test_accuracy, data.val_acc


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
        test_pr(data, model, "test")


def run_gnn_cont(filename, dir_prefix="../", lr=0.0001, exp_num=0, **kwargs):
    # ----------GNN------------------------------
    dataset = PRDataset(filename=filename)
    data_list = dataset.process()
    epochs = [30]
    for i, data in enumerate(data_list):
        model = GNNModel(data.n_teams, **kwargs)
        print("GNN model")
        continuous_evaluation(data, model, epochs[0],lr=lr, batch_size=9)
        test_cont(data, model, data.data_test, "test")
        print("accuracy on testing data is: {}".format(data.test_accuracy))
        file = outfile.format(pickle_dir.format(dir_prefix), i, exp_num, "pickle")
        data_to_save = {"data": data, "model": model, "epochs": epochs}
        save_to_pickle(file, data_to_save)
        return data.test_accuracy, data.val_acc

def run_exist_model(model_file, dir_prefix="../", lr=0.0001, exp_num=0, **kwargs):
    m = load_from_pickle(model_file)
    data = m["data"]
    model = m["model"]
    epochs = [60]
    model.eval()
    test_cont(data, model, data.data_test, "test")
    print("accuracy on testing data is: {}".format(data.test_accuracy))
    continuous_evaluation(data, model, epochs[0], lr=lr, batch_size=9, mode="test")

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
        train_cont(data, matches, model, epochs=2000, lr=0.01, print_info=True)

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


def confusion_matrix(model_file):
    from sklearn import metrics
    m = load_from_pickle(model_file)
    data = m["data"]
    model = m["model"]
    y_pred, y_true = predict(data, model, data.data_test)
    y_pred, y_true = y_pred.numpy().astype('str'), y_true.numpy().astype('str')
    y_pred[np.where(y_pred == '0')], y_true[np.where(y_true == '0')] = "home_loss", "home_loss"
    y_pred[np.where(y_pred == '1')], y_true[np.where(y_true == '1')] = "draw", "draw"
    y_pred[np.where(y_pred == '2')], y_true[np.where(y_true == '2')] = "home_win", "home_win"

    print(metrics.confusion_matrix(y_true, y_pred))
    print(metrics.classification_report(y_true, y_pred, digits=3))


if __name__ == '__main__':
    # 0:Flat, 1:PageRank, 2: GNN, 3: visualization, 4: grid search on gnn, 5: gnn cont, 6: LSTM, 7: gnn batched,
    # 8: flat cont, 9: vis cont, 10: vis embedding, 11: run_exist, 12: confusion matrix
    model_id = 9
    exp_num = "0"
    filename = "../data/GER1_2001.csv"
    # filename = "../data/0_test.csv"
    # filename = "../data/mini_data.csv"
    filename = "../data/GER1_all.csv"
    # filename = "../data/GER_second_half.csv"
    # filename = "../data/BRA1_all.csv"
    # filename = "../data/NHL.csv"

    # outfile = "{}data_{}_model_{}.{}"
    # pickle_dir = "../data/models/"
    # images_dir = "../data/img/"
    grid_search_file = "../data/grid_search_results.txt"

    if model_id == 0:
        run_cut_flat_model(filename)
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
    elif model_id == 8:
        run_flat_cont(filename)
    elif model_id == 9:
        file = "../data_0_model_154.pickle"
        data = load_from_pickle(file)
        file_to_save = outfile.format(images_dir.format(dir_prefix), 0, exp_num, "png")
        visualize_cont_eval(data["data"], file_to_save)
    elif model_id == 10:
        file = "../data_0_model_81.pickle"
        data = load_from_pickle(file)
        file_to_save = outfile.format(images_dir.format(dir_prefix), 0, exp_num, "png")
        visualize_embedding(data, file_to_save, conv=True)
    elif model_id == 3:
        file = outfile.format(pickle_dir.format(dir_prefix), 0, exp_num, "pickle")
        data = load_from_pickle(file)
        visualize_acc_loss(data["data"], data["epochs"],
                           outfile.format(images_dir.format(dir_prefix), 0, exp_num, "png"))
    elif model_id == 11:
        run_exist_model("../data_0_model_81.pickle")
    elif model_id == 12:
        confusion_matrix("../data_0_model_135.pickle")