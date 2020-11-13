target_dim = 3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from utils import update_win_lose_network, create_edge_index, update_node_time, calculate_node_weight, update_edge_time, calculate_edge_weight


def train_gnn_model(data, model, epochs=100, lr=0.001, dataset="train", print_info=False):
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
            home, away, label = matches.iloc[j]['home_team'], matches.iloc[j]['away_team'], \
                                labels[j]
                                # data.matches[0].iloc[j]['lwd']

            if j > 0:
                update_win_lose_network(data.win_lose_network, matches.iloc[j - 1])
            create_edge_index(data, home, away, matches.iloc[j]['lwd'])
            calculate_edge_weight(data)
            # calculate_node_weight(data, j, data.matches[0].shape[0])
            outputs = model(data, home, away)
            # update_node_time(data, j)
            update_edge_time(data, home, away)
            data.curr_time += 1

            loss = criterion(outputs, label.to(torch.float))
            loss.backward()
            optimizer.step()

            # print statistics
            loss_value += loss.item()
            # if j % 10 == 9:  # print every 100 mini-batches
            #     print(j)
            #     print('[%data, %5d] loss: %.3f '
            #           # 'accuracy: %.3f'
            #           %
            #           (epoch + 1, j + 1, running_loss / 100
            #            # , test_model(data, model)
            #            ))
        # TODO: Add validation data
        data.curr_time -= matches.shape[0]
        data.running_accuracy.append(test_gnn_model(data, model, "val"))
        data.running_loss.append(loss_value)
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        if epoch % 50 == 49:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8
        if print_info:
            print('[%d] Accuracy:  %.5f, loss: %.5f' % (epoch, data.running_accuracy[-1], loss_value))
    update_win_lose_network(data.win_lose_network, matches.iloc[j])
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
        labels = torch.nn.functional.one_hot(torch.tensor(np.int64(matches['lwd'])).reshape(-1, 1),
                                             num_classes=len(np.unique(np.int64(matches['lwd']))))

        for j in range(matches.shape[0]):
            home, away, label = matches.iloc[j]['home_team'], matches.iloc[j]['away_team'], \
                                labels[j]
            create_edge_index(data, home, away, matches.iloc[j]['lwd'])
            calculate_edge_weight(data)
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


def train_pr(data, model, epochs=1):
    data = list(data)
    lr_counter = 0
    c = 20

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # criterion = nn.PoissonNLLLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for param in model.parameters():
        param.requires_grad = False
    for epoch in range(epochs):
        running_loss = 0.0
        for i, d in enumerate(data):
            # get the inputs; data is a list of [inputs, labels]
            # home, away, labels = d

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # outputs = model(home, away)
            # labels = torch.nn.functional.one_hot(torch.tensor(np.int64(d.matches[0][:, 10]).reshape(-1, 1)),
            #                                      num_classes=target_dim)
            for j in range(d.matches[0].shape[0]):
                # lr_counter += 1

                home, away, label = d.matches[0].iloc[j]['home_team'], d.matches[0].iloc[j]['away_team'], d.matches[0].iloc[j][ 'lwd']#labels[j]
                if j > 0:
                    update_win_lose_network(d.win_lose_network[0], d.matches[0].iloc[j - 1])
                create_edge_index(d, home, away, label)
                calculate_node_weight(d, j, d.matches[0].shape[0])
                model(d, home, away)
                update_node_time(d, j)

    print('Finished Training')
    print('Accuracy of the network on the %s data: %.5f %%' % ("training",
                                                               test_pr(data, model)))

def test_pr(data, model, data_type=None):
    correct = 0
    total = 0
    predictions = []
    pred_index = []
    for param in model.parameters():
        param.requires_grad = False
    with torch.no_grad():
        for d in data:
            # labels = torch.nn.functional.one_hot(torch.tensor(np.int64(d.matches[0][:, 10]).reshape(-1,1)))
            for j in range(d.matches[0].shape[0]):
                home, away, label = d.matches[0].iloc[j]['home_team'], d.matches[0].iloc[j]['away_team'], \
                                    d.matches[0].iloc[j]['lwd']
                outputs = int(model(d, home, away, training=False) + 1)

                predictions.append(outputs)

                # _, predicted = torch.max(outputs.data, 1)
                # label = d.matches[0][j, 10]
                total += 1
                correct += int(outputs == label)
                pred_index.append(outputs)

    if data_type is not None:
        print('Accuracy of the network on the %s data: %.5f %%' % (data_type,
                                                               100 * correct / total))
    return 100 * correct / total


def train_flat_model(data, model, epochs=100):
    data = list(data)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.PoissonNLLLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    for epoch in range(epochs):

        running_loss = 0.0
        for i, d in enumerate(data):
            # get the inputs; data is a list of [inputs, labels]
            # home, away, labels = d

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # outputs = model(home, away)

            # labels = torch.nn.functional.one_hot(torch.tensor(np.int64(d.matches[0]['lwd']).reshape(-1, 1)),
            #                                      num_classes=d.n_teams[0])

            home, away, label = d# labels[j]
            outputs = model(home, away)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f '
                      # 'accuracy: %.3f'
                      %
                      (epoch + 1, i + 1, running_loss / 100
                       # , test_model(data, model)
                       ))
                running_loss = 0.0

    print('Finished Training')
    print('Accuracy of the network on the %s data: %.5f %%' % ("training",
                                                               test_flat_model(data, model)))


def test_flat_model(data, model, data_type=None):
    correct = 0
    total = 0

    with torch.no_grad():
        for d in data:
            home, away, labels = d
            outputs = model(home, away)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if data_type is not None:
        print('Accuracy of the network on the %s data: %.5f %%' % (data_type,
                                                                   100 * correct / total))


    return 100 * correct / total
