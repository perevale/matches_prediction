#!/usr/bin/env python
import sys
sys.path.append(sys.argv[3]+"src")
# sys.path.append(sys.argv[3]+"data")

from main import run_gnn_model, run_gnn_cont
from utils import load_from_pickle

exp_counter = sys.argv[1]
parameters_filename = sys.argv[2]
input_filename = sys.argv[3]+"data/GER1_all.csv"
output_filename = "/home/perevale/thesis/results/results_AGNN.txt"


def to_float(tup):
    return tuple(float(i) for i in tup)


def to_int(tup):
    return tuple(int(i) for i in tup)


parameters = load_from_pickle(parameters_filename)
print("EXP:|{}| embed_dim={}, n_conv={}, act_f={}, l={} \n".
            format(exp_counter, parameters["embed_dim"], parameters["n_conv"], parameters["act_f"],
                   parameters["lr"]))
acc = run_gnn_cont(input_filename,dir_prefix=sys.argv[3], lr=to_float(parameters["lr"]), exp_num=int(exp_counter), embed_dim=int(parameters["embed_dim"]),
                    n_conv=int(parameters["n_conv"]), act_f=parameters["act_f"])
print("EXP:|{}| embed_dim={}, n_conv={}, act_f={}, l={} achieved accuracy:|{}|\n".
        format(exp_counter, parameters["embed_dim"], parameters["n_conv"], parameters["act_f"],
               parameters["lr"], acc))
with open(output_filename, "a+") as f:
    f.write("EXP:|{}| embed_dim={}, n_conv={}, act_f={}, l={} achieved accuracy:|{}|\n".
            format(exp_counter, parameters["embed_dim"], parameters["n_conv"], parameters["act_f"],
                   parameters["lr"], acc))
