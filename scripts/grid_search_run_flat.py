#!/usr/bin/env python
import sys

sys.path.append(sys.argv[3] + "src")
# sys.path.append(sys.argv[3]+"data")

from main import run_flat_cont
from utils import load_from_pickle

exp_counter = sys.argv[1]
parameters_filename = sys.argv[2]
input_filename = sys.argv[3] + "data/GER1_second_half.csv"
model_name = "Flat"
output_filename = "/home/perevale/thesis/results/results_{}.txt".format(model_name)

acc, acc_val = 0, 0


def to_float(tup):
    return tuple(float(i) for i in tup)


def to_int(tup):
    return tuple(int(i) for i in tup)


def get_results():
    results = "model|{}| EXP|{}| embed_dim|{}| n_dense|{}| dense_dims|{}| val accuracy|{}| test accuracy|{}|\n" \
        .format(model_name, exp_counter, parameters["embed_dim"], parameters["n_dense"],
                parameters["dense_dims"], acc_val, acc)
    return results

parameters = load_from_pickle(parameters_filename)

results = get_results()
print(results)

acc, acc_val = run_flat_cont(input_filename,
                             dir_prefix=sys.argv[3],
                             exp_num=int(exp_counter),
                             embed_dim=int(parameters["embed_dim"]),
                             n_dense=int(parameters["n_dense"]),
                             dense_dims=to_int(parameters["dense_dims"]))

results = get_results()
print(results)

with open(output_filename, "a+") as f:
    f.write(results)
