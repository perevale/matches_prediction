import math
import pickle
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import numpy.ma as ma
import torch
from matplotlib.lines import Line2D


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


def update_edge_index(data):
    data.edge_index = torch.tensor(np.where(~np.isnan(data.edge_time)))


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


def update_node_time(data, curr_time):
    indices = data.edge_index[:][1].numpy()
    data.node_time[indices] = curr_time


def calculate_node_weight(data, curr_time):
    data.node_weight = torch.tensor(1 - ((curr_time - data.node_time) / data.N).astype(np.float32)).reshape(-1, 1)


def update_edge_time(data, home, away, result):
    winning_team = np.array([]).astype('int64')
    losing_team = np.array([]).astype('int64')

    # home won
    np.append(winning_team, home[np.where(result == 2)[0]])
    np.append(losing_team, away[np.where(result == 2)[0]])
    # away won
    np.append(winning_team, away[np.where(result == 0)[0]])
    np.append(losing_team, home[np.where(result == 0)[0]])
    # draw
    np.append(winning_team, home[np.where(result == 1)[0]])
    np.append(winning_team, away[np.where(result == 1)[0]])
    np.append(losing_team, home[np.where(result == 1)[0]])
    np.append(losing_team, away[np.where(result == 1)[0]])

    data.edge_time[losing_team, winning_team] = int(data.curr_time)


def calculate_edge_weight(data, time_weighting="linear"):
    if len(data.edge_index) > 0:
        from_nodes = data.edge_index[0].numpy()
        to_nodes = data.edge_index[1].numpy()

        prev_edge_time = data.edge_time[from_nodes, to_nodes]
        prev_edge_time[np.isnan(prev_edge_time)] = int(data.curr_time)

        if time_weighting == "linear":
            data.edge_weight = torch.tensor(1 - ((int(data.curr_time) - prev_edge_time) / data.N)).reshape(-1, ).float()
        elif time_weighting == "exponential":
            data.edge_weight = torch.tensor(np.exp(int(data.curr_time) - prev_edge_time)).reshape(-1, )
    else:
        data.edge_weight = torch.tensor([])


def visualize_acc_loss(data, epochs, file_to_save):
    area_labels = ["Training on training data", "Training on validation data"]
    colors = ['bisque', 'powderblue', 'lime']
    epochs_total = epochs[0] + epochs[1]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    legend_elements_1 = [Line2D([0], [0], label='Accuracy on validation data'),
                         Line2D([0], [0], marker='o', color='w', label=area_labels[0],
                                markerfacecolor=colors[0], markersize=15),
                         Line2D([0], [0], marker='o', color='w', label=area_labels[1],
                                markerfacecolor=colors[1], markersize=15),
                         Line2D([0], [0], color=colors[2], label='Accuracy on test data'),
                         ]
    ax[0].axvspan(0, epochs[0] - 1, color=colors[0], alpha=0.5, lw=0)
    ax[0].axvspan(epochs[0] - 1, epochs_total - 1, color=colors[1], alpha=0.5, lw=0)
    ax[0].plot(data.running_accuracy)
    ax[0].hlines(data.test_accuracy, 0, epochs_total - 1, colors=colors[2])
    lg_0 = ax[0].legend(bbox_to_anchor=(0.2, -0.55), loc='lower left',
                        ncol=1, borderaxespad=-0.3, handles=legend_elements_1)
    ax[0].title.set_text('Accuracy')
    plt.sca(ax[0])
    ticks, ticks_labels = create_ticks(epochs)
    plt.xticks(ticks, ticks_labels)
    plt.xlabel("epochs")
    plt.ylabel("accuracy [%]")

    legend_elements_1 = [Line2D([0], [0], label='Loss on validation data'),
                         Line2D([0], [0], marker='o', color='w', label=area_labels[0],
                                markerfacecolor=colors[0], markersize=15),
                         Line2D([0], [0], marker='o', color='w', label=area_labels[1],
                                markerfacecolor=colors[1], markersize=15),
                         ]
    ax[1].axvspan(0, epochs[0] - 1, color=colors[0], alpha=0.5, lw=0)
    ax[1].axvspan(epochs[0] - 1, epochs_total - 1, color=colors[1], alpha=0.5, lw=0)
    ax[1].plot(data.running_loss)
    lg_1 = ax[1].legend(bbox_to_anchor=(0.8, -0.5), loc='lower right',
                        ncol=1, borderaxespad=-0.3, handles=legend_elements_1)
    ax[1].title.set_text('Loss')
    plt.sca(ax[1])
    ticks, ticks_labels = create_ticks(epochs)
    plt.xticks(ticks, ticks_labels)
    plt.xlabel("epochs")
    plt.ylabel("loss")

    fig.tight_layout()
    plt.savefig(file_to_save,
                bbox_extra_artists=(lg_0, lg_1), format='png',
                bbox_inches='tight'
                )
    # plt.show()
    plt.clf()
    plt.cla()


def create_ticks(epochs):
    epochs_total = epochs[0] + epochs[1]
    step = math.ceil(epochs_total / 10)
    separator = epochs[0] - 1
    ticks = sorted(list(set(list(range(0, epochs_total, step)) + [separator, epochs_total - 1])))

    def clear_ticks(i_to_keep):
        delete = []
        if ticks[i_to_keep] - ticks[i_to_keep - 1] < step * 0.8:
            delete.append(i_to_keep - 1)
        if i_to_keep + 1 < len(ticks) and ticks[i_to_keep + 1] - ticks[i_to_keep] < step * 0.8:
            delete.append(i_to_keep + 1)
        for index in sorted(delete, reverse=True):
            del ticks[index]

    clear_ticks(ticks.index(separator))
    clear_ticks(ticks.index(epochs_total - 1))

    ticks_labels = [t + 1 for t in ticks]

    return ticks, ticks_labels


def save_to_pickle(filename, data):
    with open(filename, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data
