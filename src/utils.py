import torch
import numpy as np


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
        return

    from_nodes = np.append(np.full((np.count_nonzero(data.win_lose_network[0][winning_team,0] == 1),), winning_team), np.array([losing_team]).reshape((1, -1)))
    to_nodes = np.append(np.array(np.where(data.win_lose_network[0][winning_team, 0] == 1)).reshape((1, -1)), np.array([winning_team]).reshape((1, -1)))
    edge_index = torch.tensor([from_nodes,to_nodes]).long()
    data.edge_index = edge_index
    return edge_index


def get_neighbour_edge_index(data, self_loops=False):
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
    new_to_nodes = np.where(data.win_lose_network[0][to_nodes, 0] == 1)
    rep_times = np.sum(data.win_lose_network[0][to_nodes, 0], axis = 1)
    # if not self_loops:
    #     new_neighbours = ma.masked_array(new_neighbours, mask=)
    new_from_nodes = np.repeat(to_nodes, rep_times)
    edge_index = torch.tensor([new_from_nodes, new_to_nodes])
    return edge_index


def update_node_time(data, curr_time):
    indeces = data.edge_index[:][1].numpy()
    data.node_time[0][indeces] = curr_time


def calculate_node_weight(data, curr_time):
    data.node_weight = torch.tensor(1 - ((curr_time - data.node_time[0]) / data.n_datapoints[0]).astype(np.float32)).reshape(-1, 1)


def update_edge_time(data, home, away, curr_time):
    data.edge_time[0][home][away] = curr_time


def calculate_edge_weight(data, curr_time):
    from_nodes = data.edge_index[0].numpy()
    to_nodes = data.edge_index[1].numpy()

    prev_edge_time = data.edge_time[0][from_nodes, to_nodes]
    prev_edge_time[np.isnan(prev_edge_time)] = curr_time

    data.edge_weight = (1 - ((curr_time - prev_edge_time) / data.n_datapoints[0])).reshape(-1, 1).float()
