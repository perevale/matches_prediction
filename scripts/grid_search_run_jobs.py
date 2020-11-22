import subprocess
import pickle


def save_to_pickle(filename, data):
    with open(filename, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


DIR = "/home/perevale/thesis/parameters"

embed_dim = [1, 2, 3, 5, 10]
n_conv = [1, 2, 3, 4]
dims = [(1, 1, 1, 1), (2, 2, 2, 2), (4, 4, 4, 4), (8, 8, 8, 8)]
lr = [(0.0001, 0.0001)]
exp_counter = 1001
for e in embed_dim:
    for n in n_conv:
        for d in dims:
            for l in lr:
                parameters = {"embed_dim": e, "n_conv": n, "conv_dims": d, "lr": l}
                filename = DIR+"/parameters_{}.pickle".format(exp_counter)
                save_to_pickle(filename, parameters)
                command = "sbatch -o out1/slurm-%j.out train.batch {} {} ".format(exp_counter, filename)
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()

                exp_counter += 1
