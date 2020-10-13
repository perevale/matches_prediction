import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score


def train_model(data, model, epochs=1):
    data = list(data)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):

        running_loss = 0.0
        for i, d in enumerate(data):
            # get the inputs; data is a list of [inputs, labels]
            # home, away, labels = d

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # outputs = model(home, away)
            for j in range(d.matches[0].shape[0]):
                labels = torch.nn.functional.one_hot(torch.tensor(np.int64(d.matches[0][:, 10]).reshape(-1,1)))
                for j in range(d.matches[0].shape[0]):
                    home, away, label = d.matches[0][j, 3], d.matches[0][j, 4], labels[j]
                    outputs = model(d, home, away)
                    loss = criterion(outputs, torch.tensor(label).reshape(1, ))
                    loss.backward()
                    optimizer.step()

                # print statistics
                running_loss += loss.item()
                if j % 100 == 99:  # print every 100 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

    print('Finished Training')
    test_model(data, model, "train")


def test_model(data, model, data_type="test"):
    correct = 0
    total = 0
    with torch.no_grad():
        for d in data:
            labels = torch.nn.functional.one_hot(torch.tensor(d.matches[0][:, 10]))
            for j in range(d.matches[0].shape[0]):
                home, away,label = d.matches[0][j, 3], d.matches[0][j, 4], labels[j]
                outputs = model(d, home, away)
                _, predicted = torch.max(outputs.data, 1)
                total += 1
                correct += (predicted == label).sum().item()

    print('Accuracy of the network on the %s data: %.5f %%' % (data_type,
                                                               100 * correct / total))


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
