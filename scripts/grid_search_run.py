#!/usr/bin/env python
import sys

sys.path.append(sys.argv[3] + "src")
# sys.path.append(sys.argv[3]+"data")

from main import run_gnn_cont
from utils import load_from_pickle

exp_counter = sys.argv[1]
parameters_filename = sys.argv[2]
input_filename = sys.argv[3] + "data/GER_second_half.csv"
input_filename = sys.argv[3] + "data/GER1_all.csv"
model_name = "GNN"
output_filename = "/home/perevale/thesis/results/results_{}.txt".format(model_name)

acc, acc_val = 0, 0


def to_float(tup):
    return tuple(float(i) for i in tup)


def to_int(tup):
    return tuple(int(i) for i in tup)


def get_results():
    results = "model|{}| EXP|{}| embed_dim|{}| n_conv|{}| conv_dims|{}| lr|{}| val accuracy|{}| test accuracy|{}|\n" \
        .format(model_name, exp_counter, parameters["embed_dim"], parameters["n_conv"],
                parameters["conv_dims"], parameters["lr"], acc_val, acc)
    return results


parameters = load_from_pickle(parameters_filename)

results = get_results()
print(results)

acc, acc_val = run_gnn_cont(input_filename,
                             dir_prefix=sys.argv[3],
                             exp_num=int(exp_counter),
                             lr=float(parameters["lr"]),
                             embed_dim=int(parameters["embed_dim"]),
                             n_conv=int(parameters["n_conv"]),
                             conv_dims=to_int(parameters["conv_dims"]))

results = get_results()
print(results)

with open(output_filename, "a+") as f:
    f.write(results)
