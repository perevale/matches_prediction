#!/usr/bin/env python
import sys
sys.path.append(sys.argv[3]+"src")
# sys.path.append(sys.argv[3]+"data")

from main import run_gnn_model
from utils import load_from_pickle

exp_counter = sys.argv[1]
parameters_filename = sys.argv[2]
input_filename = sys.argv[3]+"data/GER1_all.csv"
output_filename = "/home/perevale/thesis/results/results.txt"

parameters = load_from_pickle(parameters_filename)
acc = run_gnn_model(input_filename, parameters["lr"], exp_counter, embed_dim=parameters["embed_dim"],
                    n_conv=parameters["n_conv"], conv_dims=parameters["conv_dims"])
with open(output_filename, "a+") as f:
    f.write("EXP:[{}] embed_dim={}, n_conv={}, conv_dims={}, l={} achieved accuracy:{}\n".
            format(exp_counter, parameters["embed_dim"], parameters["n_conv"], parameters["conv_dims"],
                   parameters["lr"], acc))
