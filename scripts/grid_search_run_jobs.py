import subprocess
import pickle


def save_to_pickle(filename, data):
    with open(filename, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


DIR = "/home/perevale/thesis/parameters"

embed_dim = [3, 10]
n_conv = [1, 2, 3]
dims = [
        (1, 1, 1),
        (4, 4, 4),
        (8, 8, 8),
        (16, 16, 16),
        (64, 64, 64),
        (128, 128, 128),
        (128, 64, 32),
        (64, 32, 16),
        # (32, 16, 8),
        # (16, 8, 4),
        # (128, 32, 8),
        # (64, 32, 4),
        # (4, 64, 4),
        # (8, 128, 8),
        # (8, 128, 64),
        # (64, 32, 64)
        ]
# activations = ['relu', 'tanh','leaky']
lr = [0.001, 0.0001]
exp_counter = 201
for e in embed_dim:
    for n in n_conv:
        for d in dims:
        # for a in activations:
            for l in lr:
                parameters = {"embed_dim": e, "n_conv": n, "conv_dims":d, "lr":l}
                filename = DIR+"/parameters_{}.pickle".format(exp_counter)
                save_to_pickle(filename, parameters)
                command = "sbatch -o out_gnn/slurm-%j.out --exclude=n33 train.batch {} {} ".format(exp_counter, filename)
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()

                exp_counter += 1
