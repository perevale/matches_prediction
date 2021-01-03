import pickle
from itertools import permutations

import numpy as np
import numpy.ma as ma
import torch


def update_edge_index(data):
    """
    Update the edge indeces
    :param data: a Dataset instance associated with the model that contains the necessary data for training and the
        metadata
    :return:
    """
    data.edge_index = torch.tensor(np.where(~np.isnan(data.edge_time)))


def update_edge_time(data, home, away, result):
    """
    Updating the time of the recent matches between the teams
    :param data: a Dataset instance associated with the model that contains the necessary data for training and the
        metadata
    :param home: the label encoded names of the home teams
    :param away: the label encoded names of the away teams
    :param result: the outcomes of the matches
    :return:
    """
    winning_team = np.array([]).astype('int64')
    losing_team = np.array([]).astype('int64')

    # home won
    winning_team = np.append(winning_team, home[np.where(result == 2)[0]])
    losing_team = np.append(losing_team, away[np.where(result == 2)[0]])
    # away won
    winning_team = np.append(winning_team, away[np.where(result == 0)[0]])
    losing_team = np.append(losing_team, home[np.where(result == 0)[0]])
    # draw
    winning_team = np.append(winning_team, home[np.where(result == 1)[0]])
    winning_team = np.append(winning_team, away[np.where(result == 1)[0]])
    losing_team = np.append(losing_team, away[np.where(result == 1)[0]])
    losing_team = np.append(losing_team, home[np.where(result == 1)[0]])

    data.edge_time[losing_team, winning_team] = int(data.curr_time)


def calculate_edge_weight(data, time_weighing="linear"):
    """
    Compute the edge weight based on the recency of the last match between the teams
    :param data: a Dataset instance associated with the model that contains the necessary data for training and the
        metadata
    :param time_weighing: the type of weighing: linear or exponential
    :return:
    """
    if len(data.edge_index) > 0:
        from_nodes = data.edge_index[0].numpy()
        to_nodes = data.edge_index[1].numpy()

        prev_edge_time = data.edge_time[from_nodes, to_nodes]
        prev_edge_time[np.isnan(prev_edge_time)] = int(data.curr_time)

        if time_weighing == "linear":
            data.edge_weight = torch.tensor(1 - ((int(data.curr_time) - prev_edge_time) / data.N)).reshape(-1, ).float()
        elif time_weighing == "exponential":
            data.edge_weight = torch.tensor(np.exp( - int(data.curr_time) - prev_edge_time)).reshape(-1, ).float()
    else:
        data.edge_weight = torch.tensor([])


def save_to_pickle(filename, data):
    """
    Saving data to pickle
    """
    with open(filename, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pickle(filename):
    """
    Loading data from pickle
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data


def compute_embedding(data, model, conv=True):
    """
    Computation of embedding for the visualization
    :param data: a Dataset instance associated with the model that contains the necessary data for training and the
        metadata
    :param model: the model
    :param conv: a boolean indicator if the model has any convolutional layers that influence the embedding
        representation
    :return:
    """
    x = torch.tensor(list(range(data.n_teams)))
    x = model.embedding(x)
    if conv:
        for layer in model.conv_layers:
            x=model.activation(layer(x, data.edge_index, data.edge_weight))
    return x

# _______________________Unused functions___________________________
def save_keras_model(filename, model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename.format("json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filename.format("h5"))


def load_keras_model(filename):
    # from tensorflow.python.keras.models import model_from_json
    # load json and create model
    json_file = open(filename.format("json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filename.format("h5"))


def update_node_time(data, curr_time):
    indices = data.edge_index[:][1].numpy()
    data.node_time[indices] = curr_time


def calculate_node_weight(data, curr_time):
    data.node_weight = torch.tensor(1 - ((curr_time - data.node_time) / data.N).astype(np.float32)).reshape(-1, 1)


def update_win_lose_network(win_lose_network, record):
    winning_team = record.home_team
    losing_team = record.away_team
    if record.lwd == 0:
        winning_team = record.away_team
        losing_team = record.home_team
    elif record.lwd == 1:
        return

    # won = 1, lost = 0
    win_lose_network[winning_team, 1, losing_team] = 1  # winning team won over losing team
    win_lose_network[losing_team, 0, winning_team] = 1  # losing team lost to the winning team


def create_test_edge_index(data):
    data.edge_index = torch.tensor(list(permutations(list(range(data.n_teams)), 2)), dtype=torch.long).t().contiguous()


def get_neighbour_edge_index(data, self_loops=False):
    from_nodes = data.edge_index[0].numpy()
    to_nodes = data.edge_index[1].numpy()

    # new_from_nodes = []
    # new_to_nodes = []
    # for node in to_nodes:
    #     new_neighbours = data.win_lose_network[0][node]['lost']
    #     if not self_loops:
    #         new_neighbours -= to_nodes
    #     new_from_nodes += [node for _ in range(len(new_neighbours))]
    #     new_to_nodes += list(new_neighbours)
    # edge_index = torch.tensor([new_from_nodes, new_from_nodes])

    # --------------------------------
    if not self_loops and len(to_nodes) > 0:
        mask = np.zeros(data.n_teams)
        mask[np.unique(np.append(from_nodes, to_nodes))] = 1
        mask = np.tile(mask, (len(to_nodes), 1))
        masked_data = ma.masked_array(data.win_lose_network[to_nodes, 0], mask=mask)
        new_to_nodes = np.where(masked_data == 1)[1]
        rep_times = np.sum(masked_data, axis=1).astype(int)
    else:
        new_to_nodes = np.where(data.win_lose_network[to_nodes, 0] == 1)[1]
        rep_times = np.sum(data.win_lose_network[to_nodes, 0], axis=1).astype(int)
    if len(new_to_nodes) > 0:
        new_from_nodes = np.repeat(to_nodes, rep_times)
        edge_index = torch.tensor([new_from_nodes, new_to_nodes])
        return edge_index
    else:
        return torch.tensor([])


def create_edge_index(data, home, away, result):
    # if result == 2:
    #     data.edge_index = torch.tensor([
    #         [home for i in range(len(data.win_lose_network[0][home]['lost']))] + [away],
    #         list(data.win_lose_network[0][home]['lost']) + [home]
    #     ])
    # elif result == 0:
    #     data.edge_index = torch.tensor([[away for i in range(len(data.win_lose_network[0][away]['lost']))] + [home],
    #                                     list(data.win_lose_network[0][away]['lost']) + [away]
    #                                     ])

    winning_team = home
    losing_team = away
    if result == 0:
        winning_team = away
        losing_team = home
    elif result == 1:
        data.edge_index = torch.tensor([[home, away], [away, home]])
        data.edge_index = get_neighbour_edge_index(data)
        if len(data.edge_index) == 0:
            data.edge_index = torch.tensor([[home, away], [away, home]])
        return

    from_nodes = np.append(np.full((np.count_nonzero(data.win_lose_network[winning_team, 0] == 1),), winning_team),
                           np.array([losing_team]).reshape((1, -1)))
    to_nodes = np.append(np.array(np.where(data.win_lose_network[winning_team, 0] == 1)).reshape((1, -1)),
                         np.array([winning_team]).reshape((1, -1)))
    edge_index = torch.tensor([from_nodes, to_nodes]).long()
    data.edge_index = edge_index
    return edge_index
