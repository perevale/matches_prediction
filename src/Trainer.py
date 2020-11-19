import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from utils import update_win_lose_network, create_edge_index, update_node_time, calculate_node_weight, update_edge_time, calculate_edge_weight, create_test_edge_index, update_edge_index

target_dim = 3


def continuous_evaluation(data, model, epochs=100, lr=0.001, dataset="train", print_info=True, model_name="gnn"):
    print("Continuous evaluation")
    train_function = train_cont_gnn
    test_function = test_cont_gnn

    if model_name == "flat":
        return

    matches = data.matches.append(data.data_val, ignore_index=True)
    matches = matches.append(data.data_test, ignore_index=True)

    for i in range(matches.shape[0]):
        home, away, result = matches.iloc[i]['home_team'], matches.iloc[i]['away_team'], \
                             matches.iloc[i]['lwd']
        test_function(data, model, home, away, result)
        train_function(data, matches.head(i+1), model, epochs, lr)


def train_cont_gnn(data, matches, model, epochs=100, lr=0.001):
    criterion = nn.PoissonNLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    running_loss = []
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

            update_edge_time(data, home, away)
            update_edge_index(data, home, away, result)
            calculate_edge_weight(data)
            data.curr_time += 1

        data.curr_time -= matches.shape[0]
        running_loss.append(loss_value)
        if epoch % 50 == 49:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8
    data.running_loss.append(sum(running_loss)/len(running_loss))


def test_cont_gnn(data, model, home, away, label):
    if len(data.edge_index) > 0:
        with torch.no_grad():
            outputs = model(data, home, away)

            _, predicted = torch.max(outputs.data, 1)
            data.running_accuracy.append(int((predicted == label).sum().item()))
    else: data.running_accuracy.append(int(False))

def train_gnn_model(data, model, epochs=100, lr=0.001, dataset="train", print_info=True):
    matches = data.matches
    if dataset == "val":
        matches = data.data_val

    criterion = nn.CrossEntropyLoss()
    criterion = nn.PoissonNLLLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):

        loss_value = 0.0
        # get the inputs; data is a list of [inputs, labels]
        # home, away, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # outputs = model(home, away)

        labels = torch.nn.functional.one_hot(torch.tensor(np.int64(matches['lwd'])).reshape(-1,1), num_classes=len(np.unique(np.int64(matches['lwd']))))
        for j in range(matches.shape[0]):
            home, away, result = matches.iloc[j]['home_team'], matches.iloc[j]['away_team'], \
                     matches.iloc[j]['lwd']
            label = labels[j]
                                # data.matches[0].iloc[j]['lwd']

            # create_edge_index(data, home, away, matches.iloc[j]['lwd'])
            # calculate_node_weight(data, j, data.matches[0].shape[0])
            if len(data.edge_index) > 0:
                outputs = model(data, home, away)
                # calculate_node_weight(data, j, data.matches[0].shape[0])

                loss = criterion(outputs, label.to(torch.float))
                loss.backward()
                optimizer.step()

                loss_value += loss.item()

            update_edge_time(data, home, away)
            # update_win_lose_network(data.win_lose_network, matches.iloc[j])
            update_edge_index(data, home, away, result)
            calculate_edge_weight(data)
            # update_node_time(data, j)
            data.curr_time += 1

        data.curr_time -= matches.shape[0]
        data.running_accuracy.append(test_gnn_model(data, model, "val"))
        data.running_loss.append(loss_value)
        if epoch % 50 == 49:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8
        if print_info:
            print('[%d] Accuracy:  %.5f, loss: %.5f' % (epoch, data.running_accuracy[-1], loss_value))
    # update_win_lose_network(data.win_lose_network, matches.iloc[j])
    if print_info:
        print('Finished training on {} data'.format(dataset))

    # plot_accuracy(running_accuracy)

def test_gnn_model(data, model, data_type="val"):
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


def correct_by_class(data, model):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for d in data:
            home, away, labels = d
            outputs = model(home, away)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            i, 100 * class_correct[i] / class_total[i]))


def evaluate(data, model):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for d in data:
            for j in range(d.matches[0].shape[0]):
                home, away, label = d.matches[0][j, 3], d.matches[0][j, 4], d.matches[0][j, 10]
                pred = model(d, home, away).numpy()

                # pred = model(data).numpy()

                # label = labels.numpy()
                predictions.append(pred)
                labels.append(label)

    y_pred = torch.FloatTensor(predictions).reshape(-1, 3)
    row_sums = torch.sum(y_pred, 1)  # normalization
    row_sums = row_sums.repeat(1, 3)  # expand to same size as out
    y_pred = torch.div(y_pred, row_sums)

    # predictions = np.hstack(predictions)
    # labels = np.hstack(labels)

    return roc_auc_score(labels, y_pred, multi_class='ovo')


def train_pr(data, model):
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

            # # forward + backward + optimize
            # # outputs = model(home, away)
            # # labels = torch.nn.functional.one_hot(torch.tensor(np.int64(d.matches[0][:, 10]).reshape(-1, 1)),
            # #                                      num_classes=target_dim)
            # for j in range(d.matches[0].shape[0]):
            #     # lr_counter += 1
            #
            #     home, away, label = d.matches[0].iloc[j]['home_team'], d.matches[0].iloc[j]['away_team'], d.matches[0].iloc[j][ 'lwd']#labels[j]
            #     if j > 0:
            #         update_win_lose_network(d.win_lose_network[0], d.matches[0].iloc[j - 1])
            #     create_edge_index(d, home, away, label)
            #     calculate_node_weight(d, j, d.matches[0].shape[0])
            #     model(d, home, away)
            #     update_node_time(d, j)

    print('Finished Training')
    print('Accuracy of the network on the %s data: %.5f %%' % ("training",
                                                               test_pr(data, model)))

def test_pr(data, model):
    correct = 0
    total = 0
    with torch.no_grad():
        matches = data.data_val
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

    # print('Accuracy of the network on the %s data: %.5f %%' % (data_type,
    #                                                            100 * correct / total))
    accuracy = 100 * correct / total
    return accuracy


def train_flat_model(data, model, epochs=100, lr=0.001, dataset="train", print_info=False):
    matches = data.matches
    if dataset == "val":
        matches = data.data_val

    criterion = nn.CrossEntropyLoss()
    criterion = nn.PoissonNLLLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    for epoch in range(epochs):

        loss_value = 0.0
        labels = torch.nn.functional.one_hot(torch.tensor(np.int64(matches['lwd'])).reshape(-1, 1),
                                             num_classes=len(np.unique(np.int64(matches['lwd']))))
        for j in range(matches.shape[0]):
            home, away, label = matches.iloc[j]['home_team'], matches.iloc[j]['away_team'], \
                                labels[j]
            optimizer.zero_grad()

            outputs = model(home, away)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            data.running_accuracy.append(test_flat_model(data, model, "val"))
            data.running_loss.append(loss_value)

        if print_info:
            print('[%d] Accuracy:  %.5f, loss: %.5f' % (epoch, data.running_accuracy[-1], loss_value))
    if print_info:
        print('Finished training on {} data'.format(dataset))


def test_flat_model(data, model, data_type=None):
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

    accuracy = 100 * correct / total
    if data_type == "test":
        data.test_accuracy = accuracy
    return accuracy
