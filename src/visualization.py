import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import numpy as np
from utils import compute_embedding

def visualize_acc_loss(data, epochs, file_to_save):
    """
    Visualize the training, validation and testing accuracy and loss for fixed training
    :param data: input data
    :param epochs:
    :param file_to_save:
    :return:
    """
    area_labels = ["Training on training data", "Training on validation data"]
    colors = ['bisque', 'powderblue', 'lime']
    epochs_total = epochs[0] + epochs[1]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    legend_elements_0 = [Line2D([0], [0], label='Accuracy on validation data'),
                         Line2D([0], [0], marker='o', color='w', label=area_labels[0],
                                markerfacecolor=colors[0], markersize=15),
                         Line2D([0], [0], marker='o', color='w', label=area_labels[1],
                                markerfacecolor=colors[1], markersize=15),
                         Line2D([0], [0], color=colors[2], label='Accuracy on test data'),
                         ]
    ax[0].axvspan(0, epochs[0] - 1, color=colors[0], alpha=0.5, lw=0)
    ax[0].axvspan(epochs[0] - 1, epochs_total - 1, color=colors[1], alpha=0.5, lw=0)
    ax[0].plot(data.val_accuracy)
    ax[0].hlines(data.test_accuracy, 0, epochs_total - 1, colors=colors[2])
    lg_0 = ax[0].legend(bbox_to_anchor=(0.2, -0.55), loc='lower left',
                        ncol=1, borderaxespad=-0.3, handles=legend_elements_0)
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
    ax[1].plot(data.train_loss)
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
    """
    A util function for creating non-clashing ticks
    :param epochs:
    :return: ticks and tick labels
    """
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


def visualize_cont_eval(data, file_to_save):
    """
    Visualize the training, validation and testing accuracy and loss for continuous evaluation
    :param data:
    :param file_to_save:
    :return:
    """
    colors = ['gold', 'royalblue', 'lime', 'silver']
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    legend_elements_0 = [Line2D([0], [0], label='Accuracy on training data', color=colors[0]),
                         Line2D([0], [0], label='Accuracy on validation data', color=colors[1]),
                         Line2D([0], [0], color=colors[2], label='Accuracy on testing data'),
                         # Line2D([0], [0], color=colors[3], label='Average accuracy on validation data', ls='--')
                         ]

    ax[0].plot([i*100 for i in data.train_accuracy], color=colors[0])
    ax[0].plot([i*100 for i in data.val_accuracy], color=colors[1])
    ax[0].hlines(data.test_accuracy*100, 0, len(data.train_accuracy), colors=colors[2])
    # ax[0].hlines(data.val_acc*100, 0, len(data.train_accuracy), ls='--', colors=colors[3])
    lg_0 = ax[0].legend(bbox_to_anchor=(0.2, -0.55), loc='lower left',
                        ncol=1, borderaxespad=-0.3, handles=legend_elements_0)
    ax[0].title.set_text('Accuracy')
    plt.sca(ax[0])
    # ticks, ticks_labels = create_ticks(epochs)
    # plt.xticks(ticks, ticks_labels)
    plt.xlabel("Time ticks")
    plt.ylabel("accuracy [%]")

    legend_elements_1 = [Line2D([0], [0], label='Loss on training data', color=colors[0]),
                         Line2D([0], [0], label='Loss on validation data', color=colors[1])
                         ]
    ax[1].plot(data.train_loss, color=colors[0])
    ax[1].plot(data.val_loss, color=colors[1])
    lg_1 = ax[1].legend(bbox_to_anchor=(0.8, -0.5), loc='lower right',
                        ncol=1, borderaxespad=-0.3, handles=legend_elements_1)
    ax[1].title.set_text('Loss')
    plt.sca(ax[1])
    plt.xlabel("Time ticks")
    plt.ylabel("loss")
    plt.yscale("log")

    fig.tight_layout()
    plt.savefig(file_to_save,
                format='png',
                bbox_inches='tight'
                )
    plt.show()
    plt.clf()
    plt.cla()



def visualize_embedding(data, file_to_save, conv=True):
    """
    Embedding visualization using PCA
    :param data:
    :param file_to_save:
    :param conv:
    :return:
    """
    from adjustText import adjust_text
    x = compute_embedding(data['data'], data['model'], conv).detach().numpy()
    pca = PCA(n_components=min(2, x.shape[1]))
    pca_result = pca.fit_transform(x)
    # plt.figure(figsize=(16, 10))
    if pca_result.shape[1]>1:
        plt.scatter(pca_result[:, 0], pca_result[:, 1])
        texts = []
        for i, team in enumerate(data['data'].teams_enc['teams']):
            texts.append(plt.text(pca_result[i, 0], pca_result[i, 1],team))
            # plt.annotate(team, xy=(pca_result[i, 0], pca_result[i, 1]), size=8)
        adjust_text(texts,arrowprops=dict(arrowstyle='-', color='red'))
    else:
        a = x[np.where(x[:, 0])]
        plt.scatter(a, np.array([1.0 for i in a]))
        for i, team in enumerate(data['data'].teams_enc['teams'][np.where(x[:, 0])[0]]):
            plt.annotate(team, xy=(a[i, 0], 0.96+0.002*i), size=8)
    plt.show()
