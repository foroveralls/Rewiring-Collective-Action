# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:27:39 2022

@author: everall
"""

# ilona presenting first week of november
# 3 November
# %%
import sys
import operator
import numpy as np
import pandas as pd
import seaborn as sns
from statistics import stdev, median, mean
import matplotlib.pyplot as plt
import os as os
import inspect
import itertools
import math
import re

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir)


# %% import data and joining together

file_extension = "csv"
parameter = "rewiring"


# stipulating regex pattern to get the parameter value from the file name string
def use_regex(input_text):
    pattern = re.compile(r"\d\.\d", re.IGNORECASE)

    return pattern.search(input_text)


file_list = os.listdir("./Output")

# get list of relevant output files
file_list = list(filter(
    lambda x: file_extension in x and parameter in x, os.listdir("./Output")))

# otherwise define it manually
# file list =

joined = []

for i in file_list:
    abs_path = os.path.join(currentdir, i)

    try:
        value = use_regex(i).group(0)
    except(AttributeError):
        print("Error: no pattern match found or file format wrong, value set to 0")
        value = 0.0

    vals = np.loadtxt(f"./Output/{i}", delimiter=",")[:, 0]

    to_append = pd.DataFrame(vals, columns=["state"])
    to_append["param"] = value

    joined.append(to_append)

runs_array = pd.concat(joined)
trajec_melt = runs_array.melt(id_vars="param", value_vars="state")


# %% convergence checks defining functions

# estimate_q taken from Scientific computing with python caam
def estimate_q_np(eps):
    """
    estimate rate of convergence q from sequence esp
    """
    x = np.arange(len(eps)-1)
    y = np.log(np.abs(np.diff(np.log(eps))))
    line = np.polyfit(x, y, 1)  # fit degree 1 polynomial
    q = np.exp(line[0])  # find q
    return q, line, y


def estimate_q(eps, loc=None, regwin=10):

    def step(x_i, x):

        x_i, x = np.log([x_i, x])

        y = np.log(np.abs(x_i - x))

        return y

    x = np.arange(len(eps) - 1)
    y = []

    if loc != None:
        # regression window
        x = x[loc-regwin: loc+regwin+1]
        print("estimating q around loc")

    for i in x:
        y.append(step(eps[i+1], eps[i]))

    y, x = np.asarray([y, x])

    idx = np.isfinite(x) & np.isfinite(y)
    line = np.polyfit(x[idx], y[idx], 1)  # fit degree 1 polynomial
    q = np.exp(line[0])  # find q

    return q, line, y, x

#test = estimate_q(eps_trunc)


# from original script
def find_root(seq):

    root = 0
    index = 0
    for i in seq:
        if abs(i) < abs(seq[root]):
            root = index
        index = index + 1

    return root


def calc_eps_diff(x_n, x_n_1):
    eps_k = abs(x_n-x_n_1)
    return eps_k


def calc_eps_trunc(x_n, x_star):
    eps_k = abs(x_n-x_star)
    return eps_k


# %% calculating epsilon for trajectories rates

trajec_eps = trajec_melt.groupby("param")

#make a copy so we don't break anything
trajec_melt_eps = trajec_melt.copy()

for group_name, trajec in trajec_eps:
    
    root_i = trajec.index[-1]
    print(trajec_melt_eps.loc[root_i, "value"])
    root = trajec_melt_eps.loc[root_i, "value"]
 
    
    for i in trajec.index:
        trajec_melt_eps.loc[i, "value"] = calc_eps_trunc(trajec.loc[i, "value"], root)
    
    


# for i in range((len(seq))):
#     eps.append(calc_eps_diff(seq[i], seq[i-1]))


# eps_trunc = []


# for i in range((len(seq)-1)):
#     root = seq[-1]
#     #ep, ep_n_i = list(itertools.starmap(calc_eps_trunc, [(seq[i], root), (seq[i+1], root)]))
#     ep = calc_eps_trunc(seq[i], root)
#     eps_trunc.append(ep)


# %% plotting functions

def epsilon_plot(epsilons):
    plt.semilogy(epsilons)
    plt.xlabel('k')
    plt.ylabel(r"$\epsilon$")
    plt.title("q = {} convergence".format(estimate_q(epsilons)[0]))
    plt.show()


def reg_plot(epsilon_diffs, loc=None):

    q, vals, y, x = estimate_q(epsilon_diffs, loc)

    #xs = list(range(len(y)))
    xs = x

    trendpoly = np.poly1d(vals)

    fig, ax = plt.subplots()
    ax.plot(xs, y)
    ax.plot(xs, trendpoly(xs))
    # ax.set_yscale('log')
    plt.xlabel('k')
    plt.ylabel(r"$\delta_epsilon$")
    plt.title("q = {} convergence".format(q))
    plt.show()
# %% plotting


for i in file_list:
     

    # checking sequence
plt.plot(seq)

# error plot
epsilon_plot(eps_trunc)

# regression plot
reg_plot(eps_trunc, loc=find_root(seq))
