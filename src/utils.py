import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.ma as ma

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
        data.edge_index = torch.tensor([[home, away], [away,home]])
        return

    from_nodes = np.append(np.full((np.count_nonzero(data.win_lose_network[winning_team,0] == 1),), winning_team), np.array([losing_team]).reshape((1, -1)))
    to_nodes = np.append(np.array(np.where(data.win_lose_network[winning_team, 0] == 1)).reshape((1, -1)), np.array([winning_team]).reshape((1, -1)))
    edge_index = torch.tensor([from_nodes,to_nodes]).long()
    data.edge_index = edge_index
    return edge_index


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
    if not self_loops and len(to_nodes)>0:
        mask = np.zeros(data.n_teams)
        mask[np.unique(np.append(from_nodes, to_nodes))] = 1
        mask = np.tile(mask, (len(to_nodes),1))
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


def update_node_time(data, curr_time):
    indeces = data.edge_index[:][1].numpy()
    data.node_time[indeces] = curr_time


def calculate_node_weight(data, curr_time):
    data.node_weight = torch.tensor(1 - ((curr_time - data.node_time) / data.N).astype(np.float32)).reshape(-1, 1)


def update_edge_time(data, home, away):
    data.edge_time[home][away] = int(data.curr_time)


def calculate_edge_weight(data, time_weighting="linear"):
    if len(data.edge_index)>0:
        from_nodes = data.edge_index[0].numpy()
        to_nodes = data.edge_index[1].numpy()

        prev_edge_time = data.edge_time[from_nodes, to_nodes]
        prev_edge_time[np.isnan(prev_edge_time)] = int(data.curr_time)

        if time_weighting == "linear":
            data.edge_weight = torch.tensor(1 - ((int(data.curr_time) - prev_edge_time) / data.N)).reshape(-1,).float()
        elif time_weighting == "exponential":
            data.edge_weight = torch.tensor(np.exp(int(data.curr_time) - prev_edge_time)).reshape(-1,)
    else:
        data.edge_weight = torch.tensor([])

def plot_accuracy(accuracy):
    sns.set_style("darkgrid")
    plt.plot(accuracy)
    plt.show()