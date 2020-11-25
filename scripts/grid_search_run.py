#!/usr/bin/env python
import sys
sys.path.append(sys.argv[3]+"src")
# sys.path.append(sys.argv[3]+"data")

from main import run_gnn_model, run_gnn_cont
from utils import load_from_pickle

exp_counter = sys.argv[1]
parameters_filename = sys.argv[2]
input_filename = sys.argv[3]+"data/GER1_all.csv"
output_filename = "/home/perevale/thesis/results/results_GRAPH.txt"


def to_float(tup):
    return tuple(float(i) for i in tup)


def to_int(tup):
    return tuple(int(i) for i in tup)


parameters = load_from_pickle(parameters_filename)
print("EXP:|{}| embed_dim={}, n_conv={}, conv_dims={}, act_f={} \n".
            format(exp_counter, parameters["embed_dim"], parameters["n_conv"], parameters["conv_dims"],
                   parameters["act_f"]))
acc = run_gnn_cont(input_filename,dir_prefix=sys.argv[3], act_f=parameters["act_f"], exp_num=int(exp_counter), embed_dim=int(parameters["embed_dim"]),
                    n_conv=int(parameters["n_conv"]), conv_dims=to_int(parameters["conv_dims"]))
print("EXP:|{}| embed_dim={}, n_conv={}, conv_dims={}, act_f={} achieved accuracy:|{}|\n".
        format(exp_counter, parameters["embed_dim"], parameters["n_conv"], parameters["conv_dims"],
               parameters["act_f"], acc))
with open(output_filename, "a+") as f:
    f.write("EXP:|{}| embed_dim={}, n_conv={}, conv_dims={}, act_f={} achieved accuracy:|{}|\n".
            format(exp_counter, parameters["embed_dim"], parameters["n_conv"], parameters["conv_dims"],
                   parameters["act_f"], acc))