import subprocess
import pickle


def save_to_pickle(filename, data):
    with open(filename, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


DIR = "/home/perevale/thesis/parameters"

embed_dim = [1, 3, 10]
n_dense = [2, 3, 4, 5]
dims = [(64, 32, 16, 8, 4), (4, 8, 16, 32, 64), (8, 8, 8, 8, 8), (16, 16, 16, 16, 16), (64, 64, 64, 64, 64), (4, 4, 4, 4, 4)]
# activations = ['relu', 'tanh','leaky']
# lr = [(0.0001, 0.0001)]
exp_counter = 1001
for e in embed_dim:
    for n in n_dense:
        for d in dims:
        # for a in activations:
        #     for l in lr:
                parameters = {"embed_dim": e, "n_dense": n, "dense_dims":d}
                filename = DIR+"/parameters_{}.pickle".format(exp_counter)
                save_to_pickle(filename, parameters)
                command = "sbatch -o out_flat/slurm-%j.out train_flat.batch {} {} ".format(exp_counter, filename)
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()

                exp_counter += 1
