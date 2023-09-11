# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:47:22 2022

@author: everall
"""

import os
import sys

sys.path.append('C:\/Users\everall\Documents\Python\Projects\It-s_how_we_talk_that_matters')
os.chdir('C:\/Users\everall\Documents\Python\Projects\It-s_how_we_talk_that_matters')

import models 
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
#from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from copy import deepcopy
#import seaborn as sns
#import pygraphviz as pgv
from statistics import stdev, mean
import imageio
import networkx as nx
from scipy.stats import truncnorm
from itertools import repeat
from itertools import islice
import time
from pathlib import Path
import dill
import models_checks
import pandas as pd
import models_checks
import seaborn as sns
from itertools import product 


modelargs=models.getargs() 



# def check_SW(graph, n = 2):
    
    
#     def calc(graph):
    
#     nx.powerlaw_cluster_graph(no)
    
#     G1, G2 = map(calc(), [graph, ])
    

#i know this has duplicates
def timer(func, *args):
    
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    
    init = time.perf_counter()
    print(f"start time:{current_time}")
    func(*args)
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(f"end time:{current_time}")
    final = time.perf_counter()
    diff = final - init
    diff_mins, secs = divmod(diff, 60)
    print(f">> runtime = {diff_mins} minutes and {round(secs, 2)} seconds")
    return diff
    

#G = G

#checking which small world measure is quicker 
#times = [(timer(nx.sigma,x), timer(nx.omega, x)) for x in [G]]
                                

rewiring =  ["bridge", "biased", "random"] #none
properties = ["HKC", "HKSW", "MC", "MSW"]
trajecs = {}
n = 20

R_G =  nx.random_reference(nx.powerlaw_cluster_graph(models.nwsize, modelargs["degree"], modelargs["clustering"]), niter = 40)

if __name__ == '__main__':
    
    
    def test(scenario, n):

        scenario_l = [scenario]
        
        for i, v in product(scenario_l, properties):
            trajecs[f"{i}_{v}"] = []

        
        for i in np.arange(n):
            
        
            print("model start")
            model_HK = models_checks.simulate(i, modelargs, R_G)
            trajecs[f"{scenario}_HKC"] += [model_HK.clustering_diff]
            trajecs[f"{scenario}_HKSW"] += [model_HK.small_world_diff]
            
            print("model standard start ")
            model = models.simulate(i, modelargs, R_G)
            trajecs[f"{scenario}_MC"] += [model.clustering_diff]
            trajecs[f"{scenario}_MSW"] += [model.small_world_diff]
            
            print(f"{i}/{n}complete")

        return trajecs
    
    print("I'm running")
    
    def run(n):
        for i in rewiring:
            test(i, n)

    timer(run, n)

    trajecs_save = trajecs
    
# trajecs = {key: val[:50] for key, val in d_comb.items()}

#python literally sucks ass for dataframe manipulation
    props = pd.DataFrame(trajecs)
    props["id"] = np.random.rand(len(props))
    props_m = pd.wide_to_long(props, stubnames = rewiring,
                              i = "id", j = "property", suffix='\w+', sep = "_")
    
 
    props_m =props_m.reset_index()
    
    #props_m = pd.read_csv("network_checks_21_07.csv")
    
    
    props_m = pd.melt(props_m, value_name = "delta", 
                      var_name = "scenario", value_vars = rewiring, id_vars = "property")
    
    props_m["Comparison"] = np.where(props_m["property"].str.contains("C"),  "Clustering", "Small_World")  
    
    #props_m.to_csv("network_checks_n600_T5000.csv", index = False)
    
    g1 = sns.displot(col = "scenario", row = "Comparison", x = "delta", data = props_m,
                    hue = "property",  alpha = 0.7, legend = True,
                    bins = 80, kde = True)
    
    g2 = sns.displot(col = "scenario", x = "delta", data = props_m,
                    hue = "property", col_wrap = 2,  alpha = 0.7, legend = True,
                    multiple = "stack", bins = 80, )
    
        
    g3 = sns.displot(col = "scenario", x = "delta", data = props_m,
                    hue = "property", col_wrap = 2,  alpha = 0.5, legend = True, kind = "kde")
    
    #saving graphs
    graphs = [g1, g2, g3]
    for i, v in enumerate(graphs):
        v.savefig(f"../Figs/nw_analysis_g{i}_N_{models.nwsize}_tmax{models.timesteps}.png", dpi = 300, bbox_inchex = "tight")
    
    
    # plt.legend(labels = ["HK Clustering", "HK SmallWorld", "Default Clustering", "Default Small World"],
    #            bbox_to_anchor=(1.5,0.8), loc ="lower right")
    
