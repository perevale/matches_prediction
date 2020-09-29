import torch.optim as optim
import torch.nn as nn
import torch


def train_model(data, model, epochs=1):
    data = list(data)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):

        running_loss = 0.0
        for i, d in enumerate(data):
            # get the inputs; data is a list of [inputs, labels]
            home, away, labels = d
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(home, away)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')
    # test_model(data1, model, "train")


def test_model(data, model, data_type="test"):
    correct = 0
    total = 0
    with torch.no_grad():
        for d in data:
            home, away, labels = d
            outputs = model(home, away)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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