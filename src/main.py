from typing import Tuple

from DataTransformer import DataTransformer
from FlatModel import FlatModel
from Trainer import *
from GNNModel import GNNModel
from Dataset import Dataset
from SimpleGNN import PageRank
from utils import save_to_pickle, load_from_pickle
from visualization import visualize_acc_loss, visualize_cont_eval, visualize_embedding

dir_prefix = "../"
outfile = "{}data_{}_model_{}.{}"
pickle_dir = "{}data/models/"
images_dir = "{}data/img/"


def run_flat_cont(filename, dir_prefix="../", lr=0.001, exp_num=0, **kwargs):
    """
    Training a Simple ANN model with embedding using continuous evaluation. The model is then saved into pickle.
    :param filename: the name of the file with the input data
    :param dir_prefix: directory prefix for model saving
    :param lr: a learning rate for training
    :param exp_num: experiment number
    :param kwargs: additional parameters for the model
    :return:
    """
    dataset = Dataset(filename=filename)
    data_list = dataset.process()  # load and process all the data
    epochs = [30]  # number of initial epochs
    test_acc = []
    val_acc = []
    n_all_teams = 0
    for data in data_list:
        n_all_teams += data.n_teams
    model = FlatModel(n_all_teams, **kwargs)

    for i, data in enumerate(data_list):
        print("Flat model, data {}", i)
        continuous_evaluation(data, model, epochs[0],lr=lr, batch_size=9)
        test_cont(data, model, data.data_test, "test")
        print("accuracy on testing data is: {}".format(data.test_accuracy))
        file = outfile.format(pickle_dir.format(dir_prefix), i, exp_num, "pickle")
        data_to_save = {"data": data, "model": model, "epochs": epochs}
        save_to_pickle(file, data_to_save)
        test_acc.append(data.test_accuracy)
        val_acc.append(data.val_acc)

    test_accuracy = sum(test_acc)/len(test_acc)
    val_accuracy = sum(val_acc)/len(val_acc)
    file = outfile.format(pickle_dir.format(dir_prefix), "all", exp_num, "pickle")
    data_to_save = {"test_acc":test_acc, "val_acc":val_acc}
    save_to_pickle(file, data_to_save)
    return test_accuracy, val_accuracy


def run_gnn_cont(filename, dir_prefix="../", lr=0.0001, exp_num=0, **kwargs):
    """
    Training a GNN model using continuous evaluation. The model is then saved into pickle.
    :param filename: the name of the file with the input data
    :param dir_prefix: directory prefix for model saving
    :param lr: a learning rate for training
    :param exp_num: experiment number
    :param kwargs: additional parameters for the model
    :return:
    """
    dataset = Dataset(filename=filename)
    data_list = dataset.process() # load and process all the data
    epochs = [30] # number of initial epochs
    test_acc = []
    val_acc = []
    n_all_teams = 0
    for data in data_list:
        n_all_teams += data.n_teams
    model = GNNModel(n_all_teams, **kwargs)

    for i, data in enumerate(data_list):
        print("GNN model, data {}", i)
        continuous_evaluation(data, model, epochs[0],lr=lr, batch_size=9)
        test_cont(data, model, data.data_test, "test")
        print("accuracy on testing data is: {}".format(data.test_accuracy))
        file = outfile.format(pickle_dir.format(dir_prefix), i, exp_num, "pickle")
        data_to_save = {"data": data, "model": model, "epochs": epochs}
        save_to_pickle(file, data_to_save)
        test_acc.append(data.test_accuracy)
        val_acc.append(data.val_acc)

    test_accuracy = sum(test_acc) / len(test_acc)
    val_accuracy = sum(val_acc) / len(val_acc)
    file = outfile.format(pickle_dir.format(dir_prefix), "all", exp_num, "pickle")
    data_to_save = {"test_acc": test_acc, "val_acc": val_acc}
    save_to_pickle(file, data_to_save)
    return test_accuracy, val_accuracy


def run_exist_model(model_file: str, lr:float = 0.0001):
    """
    Run a trained model on the testing data and retrain the model using it (sliding testing set)
    :param model_file: input file with the saved model and the data
    :param lr: tuple of learning rates for training and validation
    :return:
    """
    m = load_from_pickle(model_file)
    data = m["data"]
    model = m["model"]
    epochs = [60]
    model.eval()
    test_cont(data, model, data.data_test, "test")
    print("accuracy on testing data is: {}".format(data.test_accuracy))
    continuous_evaluation(data, model, epochs[0], lr=lr, batch_size=9, mode="test")


def confusion_matrix(model_file):
    """
    Compute the confusion matrix of the trained model
    :param model_file: input file with the saved model and the data
    :return:
    """
    from sklearn import metrics
    m = load_from_pickle(model_file)
    data = m["data"]
    model = m["model"]
    y_pred, y_true, prob = get_predictions(data, model, data.data_test)
    y_pred, y_true = y_pred.numpy().astype('str'), y_true.numpy().astype('str')
    y_pred[np.where(y_pred == '0')], y_true[np.where(y_true == '0')] = "home_loss", "home_loss"
    y_pred[np.where(y_pred == '1')], y_true[np.where(y_true == '1')] = "draw", "draw"
    y_pred[np.where(y_pred == '2')], y_true[np.where(y_true == '2')] = "home_win", "home_win"

    print(metrics.confusion_matrix(y_true, y_pred))
    print(metrics.classification_report(y_true, y_pred, digits=3))


def calculate_rps(model_file: str):
    """
    Compute the RPS score of the trained model on testing data
    :param model_file: input file with the saved model and the data
    :return:
    """
    m = load_from_pickle(model_file)
    data = m["data"]
    model = m["model"]
    rps = get_rps(data, model, data.data_test)
    print(torch.mean(rps).item())


# _________________________Unused functions_________________________
def common_loading(filename: str) -> DataTransformer:
    """
    Printing out general information about the data set.

    :param filename: the name of the input file
    :return: instance of a DataTransformer class
    """
    dt = DataTransformer(filename)
    # home, away, label = dt.get_train_data()
    print("Information on the whole dataset: ")
    dt.print_metadata(dt.data, "whole")
    return dt


def run_pr_model(filename: str):
    """
    An early PageRank model
    :param filename: the name of the file with the input data
    :return:
    """
    dataset = Dataset(filename=filename)
    data_list = dataset.process()
    for i, data in enumerate(data_list):
        model = PageRank()
        train_pr(data, model)
        test_pr(data, model, "test")


def run_fixed_flat_model(filename: str, dir_prefix: str = "../", lr: Tuple = (0.001, 0.001), exp_num: int = 0, **kwargs):
    """
    The function trains a Simple ANN model with embedding with fixed training, validation and testing data sets.
    The model is first trained on training data, then retrained on a validation data set and then tested on tested data.
    The function is not used anymore since it does not follow the correct assumptions
    :param filename: the name of the file with the input data
    :param dir_prefix: directory prefix for model saving
    :param lr: tuple of learning rates for training and validation
    :param exp_num: experiment number
    :param kwargs: additional parameters for the model
    :return:
    """
    dataset = Dataset(filename=filename)
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


def run_fixed_gnn_model(filename: str, dir_prefix: str = "../", lr: Tuple = (0.00001, 0.0001), exp_num: int = 0, **kwargs):
    """
    The function trains a GNN model with embedding with fixed training, validation and testing data sets.
    The model is first trained on training data, then retrained on a validation data set and then tested on tested data.
    The function is not used anymore since it does not follow the correct assumptions
    :param filename: the name of the file with the input data
    :param dir_prefix: directory prefix for model saving
    :param lr: tuple of learning rates for training and validation
    :param exp_num: experiment number
    :param kwargs: additional parameters for the model
    :return:
    """
    dataset = Dataset(filename=filename)
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


if __name__ == '__main__':
    # 5: gnn cont, 8: flat cont, 11: run_exist, 9: vis cont, 10: vis embedding,
    # 12: confusion matrix, 13: rps
    # UNUSED:
    # 0:Flat, 1:PageRank, 2: GNN, 3: visualization

    function_id = 5
    exp_num = "0"
    dataset_filename = "../data/soccer_all_leagues.csv"
    dataset_filename = "../data/NHL.csv"
    dataset_filename = "../data/GER1_all.csv"
    model_filename = "../data_0_model_154.pickle"

    if function_id == 5:
        run_gnn_cont(dataset_filename)
    elif function_id == 8:
        run_flat_cont(dataset_filename)
    elif function_id == 9:
        data = load_from_pickle(model_filename)
        file_to_save = outfile.format(images_dir.format(dir_prefix), 0, exp_num, "png")
        visualize_cont_eval(data["data"], file_to_save)
    elif function_id == 10:
        data = load_from_pickle(model_filename)
        file_to_save = outfile.format(images_dir.format(dir_prefix), 0, exp_num, "png")
        visualize_embedding(data, file_to_save, conv=True)
    elif function_id == 11:
        run_exist_model(model_filename)
    elif function_id == 12:
        confusion_matrix(model_filename)
    elif function_id == 13:
        calculate_rps(model_filename)

    # _____________Unused functions________________________________________
    elif function_id == 0:
        run_fixed_flat_model(dataset_filename)
    elif function_id == 1:
        run_pr_model(dataset_filename)
    elif function_id == 2:
        run_fixed_gnn_model(dataset_filename)
    elif function_id == 3:
        file = outfile.format(pickle_dir.format(dir_prefix), 0, exp_num, "pickle")
        data = load_from_pickle(file)
        visualize_acc_loss(data["data"], data["epochs"],
                           outfile.format(images_dir.format(dir_prefix), 0, exp_num, "png"))
