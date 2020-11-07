import torch
import numpy as np


def calculate_win_lose_network(group, n_teams):
    win_draw = group[(group.lwd == 2)]  # (group.lwd == 1) |
    lose = group[(group.lwd == 0)]
    win_lose_network = [{'won': set(), 'lost': set()} for _ in range(n_teams)]
    for i in range(n_teams):
        # won
        home_team = win_draw[win_draw.home_team == i]
        win_lose_network[i]['won'].update(set(home_team.away_team))
        away_team = lose[lose.away_team == i]
        win_lose_network[i]['won'].update(set(away_team.home_team))
        # lost
        home_team = lose[lose.home_team == i]
        win_lose_network[i]['lost'].update(set(home_team.away_team))
        away_team = win_draw[win_draw.away_team == i]
        win_lose_network[i]['lost'].update(set(away_team.home_team))
    return win_lose_network


def update_win_lose_network(win_lose_network, record):
    if record.lwd == 2:  # record.lwd == 1 or

        win_lose_network[record.home_team]['won'].add(record.away_team)
        win_lose_network[record.away_team]['lost'].add(record.home_team)
    elif record.lwd == 0:
        win_lose_network[record.home_team]['lost'].add(record.away_team)
        win_lose_network[record.away_team]['won'].add(record.home_team)
    else:
        pass
    # return win_lose_network


def create_edge_index(data, home, away, result):
    if result == 2:
        data.edge_index = torch.tensor([
            [home for i in range(len(data.win_lose_network[0][home]['lost']))] + [away],
            list(data.win_lose_network[0][home]['lost']) + [home]
        ])
    elif result == 1:
        data.edge_index = torch.tensor([[away for i in range(len(data.win_lose_network[0][away]['lost']))] + [home],
                                        list(data.win_lose_network[0][away]['lost']) + [away]
                                        ])

    # np.concatenate((np.full((len(data.win_lose_network[0][home]['lost']),), home), np.array([home])), axis=0)
    # np.concatenate((list(data.win_lose_network[0][home]['lost']), np.array([away])), axis=0)


def get_neighbour_edge_index(data, self_loops=False):
    to_nodes = data.edge_index[1].numpy()

    new_from_nodes = []
    new_to_nodes = []
    for node in to_nodes:
        new_neighbours = data.win_lose_network[0][node]['lost']
        if not self_loops:
            new_neighbours -= to_nodes
        new_from_nodes += [node for _ in range(len(new_neighbours))]
        new_to_nodes += list(new_neighbours)
    edge_index = torch.tensor([new_from_nodes, new_from_nodes])
    return edge_index


def update_node_time(data, curr_time):
    indeces = data.edge_index[:][1].numpy()
    data.node_time[0][indeces] = curr_time


def calculate_node_weight(data, curr_time):
    # if data.edge_time[home,away] is not None and data.edge_time[away] is not None:

    data.node_weight = torch.tensor(1 - ((curr_time - data.node_time[0]) / data.n_datapoints[0]).astype(np.float32)).reshape(-1, 1)


def update_edge_time(data, home, away, curr_time):
    data.edge_time[0][home][away] = curr_time

def calculate_edge_weight(data, curr_time):
    from_nodes = data.edge_index[0].numpy()
    to_nodes = data.edge_index[1].numpy()

    prev_edge_time = data.edge_time[0][from_nodes, to_nodes]
    prev_edge_time[np.isnan(prev_edge_time)] = curr_time


    data.edge_weight = (1 - ((curr_time - prev_edge_time) / data.n_datapoints[0])).reshape(-1, 1).float()
