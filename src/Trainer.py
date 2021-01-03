import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
import pandas as pd
from torch_geometric.data import Data

from utils import update_win_lose_network, create_edge_index, update_node_time, \
    calculate_node_weight, update_edge_time, calculate_edge_weight, update_edge_index

log_base = 1.5
val_batches = 72
target_dim = 3


def continuous_evaluation(data: Data, model, epochs=100, lr=0.001, lr_discount=0.2, batch_size=9, mode="val"):
    """
    A gateway function for starting the training, validation and testing of the provided model using continuous
    evaluation
    :param data: a Dataset instance associated with the model that contains the necessary data for training and the
        metadata
    :param model: the model
    :param epochs: the number of epochs
    :param lr: the learning rate
    :param lr_discount: the learning rate discount if the adaptable learning rate is used
    :param batch_size: size of batches in which the model is trained
    :param mode: mode in which the function is used: "val" for validation, "test" for testing
    :return:
    """
    print("Continuous evaluation")
    train_function = train_cont
    test_function = test_cont

    if mode == "val":
        matches = data.matches.append(data.data_val, ignore_index=True)
    else:
        matches = data.data_test
        test_acc = []
        global val_batches
        val_batches = 1

    for i in range(0, matches.shape[0], batch_size):
        test_function(data, model, matches.iloc[i:i + val_batches * batch_size])
        train_start_point = max(0, i - 40 * batch_size)
        data.curr_time = train_start_point
        train_function(data,
                       # matches.head(i + batch_size),
                       matches.iloc[train_start_point:i + batch_size],
                       model,
                       # epochs,
                       epochs + int(math.log(i + 1, log_base)),
                       # lr * (1 - lr_discount) ** int(i / batch_size / 50),
                       lr,
                       batch_size)
        print("T:{}, train_loss:{:.5f}, train_acc:{:.5f}, val_loss={:.5f}, val_acc={:.5f}"
              .format(int(i / batch_size),
                      data.train_loss[-1],
                      data.train_accuracy[-1],
                      data.val_loss[-1],
                      data.val_accuracy[-1]))
        if mode == "test":
            test_acc.append(data.val_accuracy[-1])

    val_acc = data.val_accuracy[len(data.val_accuracy) - val_batches:]
    data.val_acc = sum(val_acc) / len(val_acc)

    if mode == "test":
        print(sum(test_acc) / len(test_acc))
    else:
        print(data.val_acc)


def train_cont(data: Data, model: torch.nn.Module, matches: pd.Dataframe,
               epochs:int = 100, lr: int = 0.0001, batch_size:int = 9, print_info: bool = False):
    """
    A function for training the provided model with the provided matches using continuous evaluation
    :param data: a Dataset instance associated with the model that contains the necessary data for training and the
        metadata
    :param model: the model
    :param matches: the data set
    :param epochs: the number of epochs
    :param lr: the learning rate
    :param batch_size: size of batches in which the model is trained
    :param print_info: a binary flag that indicates if the information about the training should be printed out to the
        terminal
    :return:
    """
    criterion = nn.NLLLoss()  # weight=torch.tensor([1.6,1.95,1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    running_loss = []
    running_accuracy = []
    home_win = 0
    for epoch in range(epochs):
        acc = 0
        loss_value = 0.0
        optimizer.zero_grad()
        for j in range(0, matches.shape[0], batch_size):
            home, away, result = torch.from_numpy(matches.iloc[j:j + batch_size]['home_team'].values.astype('int64')), \
                                 torch.from_numpy(matches.iloc[j:j + batch_size]['away_team'].values.astype('int64')), \
                                 torch.from_numpy(
                                     matches.iloc[j:j + batch_size]['lwd'].values.astype('int64').reshape(-1, ))
            home_win += (result == 2).sum().item()
            # label = torch.zeros(result.shape[0], target_dim).scatter_(1, torch.tensor(result), 1)  # one-hot label for loss
            outputs = model(data, home, away)
            # loss = criterion(outputs, label.to(torch.float))
            loss = criterion(outputs, result)
            loss.backward()
            optimizer.step()
            loss_value += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct = int((predicted == result).sum().item())
            running_accuracy.append(correct)
            acc += correct

            update_edge_time(data, home, away, result)
            update_edge_index(data)
            calculate_edge_weight(data)
            data.curr_time += 1

        if print_info:
            print("Epoch:{}, train_loss:{:.5f}, train_acc:{:.5f}"
                  .format(epoch, loss_value, acc / (matches.shape[0])))

        data.curr_time -= math.ceil(matches.shape[0] / batch_size)  # probably is safe to be set to 0 each epoch
        running_loss.append(loss_value)
        # if epoch % 50 == 49:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.8
    # print(home_win/(matches.shape[0] * epochs))
    data.train_loss.append(sum(running_loss) / ((matches.shape[0] / batch_size) * epochs))
    data.train_accuracy.append(sum(running_accuracy) / (matches.shape[0] * epochs))


def test_cont(data: Data, model: torch.nn.Module, matches: pd.Dataframe, mode: str = "val"):
    """
    A function for testing the provided model on the provided matches using continuous evaluation
    :param data: a Dataset instance associated with the model that contains the necessary data for training and the
        metadata
    :param model: the model
    :param matches: the data set
    :param mode: mode in which the testing function is used: "val" for validation, "test" for testing
    :return:
    """
    criterion = nn.NLLLoss()  # weight=torch.tensor([1.6,1.95,1])

    predicted, label, outputs = get_predictions(data, model, matches)

    loss = criterion(outputs, label).item()

    correct = int((predicted == label).sum().item())
    if mode == "test":
        data.test_accuracy = float(correct) / matches.shape[0]
    else:
        data.val_accuracy.append(float(correct) / matches.shape[0])
        data.val_loss.append(loss)


def get_predictions(data: Data, model: torch.nn.Module, matches: pd.Dataframe) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the predictions for the provided matches using provided model
    :param data: a Dataset instance associated with the model that contains the necessary data for training and the
        metadata
    :param model: the model
    :param matches: the data set
    :return: Tensors of predictions for each match, a ground truth label and the probabilities
    """
    outputs, label = get_probabilities(data, model, matches)
    _, predicted = torch.max(torch.exp(outputs.data), 1)
    return predicted, label, outputs


def get_probabilities(data, model, matches):
    model.eval()
    home, away, label = torch.from_numpy(matches['home_team'].values.astype('int64')), \
                        torch.from_numpy(matches['away_team'].values.astype('int64')), \
                        torch.from_numpy(matches['lwd'].values.astype('int64').reshape(-1, ))
    with torch.no_grad():
        outputs = model(data, home, away)
    model.train()
    return outputs, label


def get_rps(data: Data, model: torch.nn.Module, matches: pd.Dataframe) -> float:
    """
    Computation of PRS score for a provided model on provided data
    :param data: a Dataset instance associated with the model that contains the necessary data for training and the
        metadata
    :param model: the model
    :param matches: the data set
    :return:
    """
    outputs, label = get_probabilities(data, model, matches)
    _, predicted = torch.max(outputs.data, 1)
    o_h_label = torch.zeros(label.shape[0], target_dim).scatter_(1, label.reshape(-1, 1), 1)  # one-hot label for loss
    outputs = torch.exp(outputs)
    sub = torch.zeros((label.shape[0]))
    for i in range(target_dim):
        sub += torch.pow(torch.sum(outputs[:, :i + 1], 1) - torch.sum(o_h_label[:, :i + 1], 1), 2)
    rps = sub / (target_dim - 1)
    return rps


# ______________________Unused functions_______________________________________________
def train_pr(data, model):
    """
    Training a Page Rank model
    """
    matches = data.matches

    for param in model.parameters():
        param.requires_grad = False

    labels = torch.nn.functional.one_hot(torch.tensor(np.int64(matches['lwd'])).reshape(-1, 1),
                                         num_classes=len(np.unique(np.int64(matches['lwd']))))
    for j in range(matches.shape[0]):
        home, away, label = matches.iloc[j]['home_team'], matches.iloc[j]['away_team'], \
                            labels[j]
        # data.matches[0].iloc[j]['lwd']

        if j > 0:
            update_win_lose_network(data.win_lose_network, matches.iloc[j - 1])
        create_edge_index(data, home, away, matches.iloc[j]['lwd'])
        # calculate_edge_weight(data)
        calculate_node_weight(data, j)
        outputs = model(data, home, away)
        update_node_time(data, j)
        # update_edge_time(data, home, away)
        data.curr_time += 1

    print('Finished Training')
    print('Accuracy of the network on the %s data: %.5f %%' % ("training",
                                                               test_pr(data, model)))


def test_pr(data, model, data_type="val"):
    """
    Testing a PageRank model
    """
    correct = 0
    total = 0
    matches = data.matches
    if data_type == "val":
        matches = data.data_val
    elif data_type == "test":
        matches = data.data_test
    with torch.no_grad():

        labels = torch.nn.functional.one_hot(torch.tensor(np.int64(matches['lwd'])).reshape(-1, 1),
                                             num_classes=len(np.unique(np.int64(matches['lwd']))))

        for j in range(matches.shape[0]):
            home, away, label = matches.iloc[j]['home_team'], matches.iloc[j]['away_team'], \
                                labels[j]
            create_edge_index(data, home, away, matches.iloc[j]['lwd'])
            calculate_edge_weight(data)
            outputs = int(model(data, home, away, training=False) + 1)

            label = matches.iloc[j]['lwd']
            total += 1
            correct += int(outputs == label)

    print('Accuracy of the network on the %s data: %.5f %%' % (data_type,
                                                               100 * correct / total))
    accuracy = 100 * correct / total
    return accuracy


def train_gnn_model(data, model, epochs=100, lr=0.01, dataset="train", print_info=True):
    """
    Training the GNN model with a fixed training, validation and testing sets
    """
    matches = data.matches
    if dataset == "val":
        matches = data.data_val

    criterion = nn.CrossEntropyLoss()
    criterion = nn.PoissonNLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):

        loss_value = 0.0
        optimizer.zero_grad()
        labels = torch.nn.functional.one_hot(torch.tensor(np.int64(matches['lwd'])).reshape(-1, 1),
                                             num_classes=len(np.unique(np.int64(matches['lwd']))))
        for j in range(matches.shape[0]):
            home, away, result = matches.iloc[j]['home_team'], matches.iloc[j]['away_team'], \
                                 matches.iloc[j]['lwd']
            label = labels[j]
            if len(data.edge_index) > 0:
                outputs = model(data, home, away)
                loss = criterion(outputs, label.to(torch.float))
                loss.backward()
                optimizer.step()
                loss_value += loss.item()

            update_edge_time(data, home, away, result)
            # update_win_lose_network(data.win_lose_network, matches.iloc[j])
            update_edge_index(data)
            calculate_edge_weight(data)
            # update_node_time(data, j)
            data.curr_time += 1

        data.curr_time -= matches.shape[0]
        data.val_accuracy.append(test_gnn_model(data, model, "val"))
        data.train_loss.append(loss_value)
        if epoch % 50 == 49:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8
        if print_info:
            print('[%d] Accuracy:  %.5f, loss: %.5f' % (epoch, data.val_accuracy[-1], loss_value))
    # update_win_lose_network(data.win_lose_network, matches.iloc[j])
    if print_info:
        print('Finished training on {} data'.format(dataset))

    # plot_accuracy(running_accuracy)


def test_gnn_model(data, model, data_type="val"):
    """
        Testing the GNN model with a fixed training, validation and testing sets
    """
    correct = 0
    total = 0
    with torch.no_grad():
        matches = data.matches
        if data_type == "val":
            matches = data.data_val
        elif data_type == "test":
            matches = data.data_test
        for j in range(matches.shape[0]):
            home, away = matches.iloc[j]['home_team'], matches.iloc[j]['away_team']
            outputs = model(data, home, away)

            _, predicted = torch.max(outputs.data, 1)
            label = matches.iloc[j]['lwd']
            total += 1
            correct += (predicted == label).sum().item()

    # print('Accuracy of the network on the %s data: %.5f %%' % (data_type,
    #                                                            100 * correct / total))
    accuracy = 100 * correct / total
    if data_type == "test":
        data.test_accuracy = accuracy
    return accuracy


def train_flat_model(data, model, epochs=100, lr=0.001, dataset="train", print_info=False, batch_size=9):
    """
        Training the simple ANN with encoding model with a fixed training, validation and testing sets
    """
    matches = data.matches
    if dataset == "val":
        matches = data.data_val

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    running_loss = []
    running_accuracy = []
    for epoch in range(epochs):
        acc = 0
        loss_value = 0.0
        optimizer.zero_grad()
        for j in range(0, matches.shape[0], batch_size):
            home, away, result = torch.from_numpy(matches.iloc[j:j + batch_size]['home_team'].values.astype('int64')), \
                                 torch.from_numpy(matches.iloc[j:j + batch_size]['away_team'].values.astype('int64')), \
                                 torch.from_numpy(
                                     matches.iloc[j:j + batch_size]['lwd'].values.astype('int64').reshape(-1, ))
            # label = torch.zeros(result.shape[0], target_dim).scatter_(1, torch.tensor(result), 1)  # one-hot label for loss
            outputs = model(data, home, away)
            # loss = criterion(outputs, label.to(torch.float))
            loss = criterion(outputs, result)
            loss.backward()
            optimizer.step()
            loss_value += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct = int((predicted == result).sum().item())
            running_accuracy.append(correct)
            acc += correct

        test_flat_model(data, model, "val")
        if print_info:
            print("Epoch:{}, train_loss:{:.5f}, train_acc:{:.5f}, val_loss={:.5f}, val_acc={:.5f}"
                  .format(epoch, loss_value / (matches.shape[0] / batch_size), acc / (matches.shape[0]),
                          data.val_loss[-1],
                          data.val_accuracy[-1]
                          ))

        running_loss.append(loss_value)
        if epoch % 50 == 49:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8
    data.train_loss.append(sum(running_loss) / ((matches.shape[0] / batch_size) * epochs))
    data.train_accuracy.append(sum(running_accuracy) / (matches.shape[0] * epochs))


def test_flat_model(data, model, data_type=None):
    """
        Testing the simple ANN with encoding model with a fixed training, validation and testing sets
    """
    matches = data.matches
    if data_type == "val":
        matches = data.data_val
    elif data_type == "test":
        matches = data.data_test
    criterion = nn.NLLLoss()

    home, away, label = torch.from_numpy(matches['home_team'].values.astype('int64')), \
                        torch.from_numpy(matches['away_team'].values.astype('int64')), \
                        torch.from_numpy(matches['lwd'].values.astype('int64').reshape(-1, ))
    with torch.no_grad():
        outputs = model(data, home, away)
        loss = criterion(outputs, label).item()

        _, predicted = torch.max(outputs.data, 1)
        correct = int((predicted == label).sum().item())
        if data_type == "test":
            data.test_accuracy = float(correct) / matches.shape[0]
        else:
            data.val_accuracy.append(float(correct) / matches.shape[0])
            data.val_loss.append(loss)
