# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:17:44 2024

@author: Jordan
"""


#%%
import networkx as nx
import matplotlib.pyplot as plt
from statistics import stdev, mean
import numpy as np



#%%
#-------- drawing functions ---------



def findAvgStateInClusters(model, part):
    states = [[] for i in range(len(set(part.values())))]
   
    for n, v in part.items():
        states[v].append(model.graph.nodes[n]['agent'].state)
    clusters = []
    sd = []
    clsize = []
    for c in range(len(states)):
        clusters.append(mean(states[c]))
        clsize.append(len(states[c]))
        if(len(states[c])>1):
            sd.append(stdev(states[c]))
        else:
            sd.append(0) 
    return (clusters, sd, clsize)

def findAvgSDinClusters(model, part):
    states = [[] for i in range(len(set(part.values())))]
    for n, v in part.items():
        states[v].append(model.graph.nodes[n]['agent'].state)
    
    sd = []
    for c in range(len(states)):
        if(len(states[c])>1):
            sd.append(stdev(states[c]))
        else:
            sd.append(0)
    return sd


def reduce_grid(model):
    n=gridsize
    for i in range(n):
        for j in range(n):
            if(i!=0 and j!=0 ):
                model.graph.remove_edge(i*n+j, (i-1)*n+j-1)
            if(i!=0 and j!=(n-1)):
                model.graph.remove_edge(i*n+j, (i-1)*n+j+1)
            """
            if( i != n-1 and j!= n-1):
                weight = model.getFriendshipWeight()
                model.graph.remove_edge(i*n+j, (i+1)*n+j+1, weight = weight)
            if(j != 0 and i != n-i):
                weight = model.getFriendshipWeight()
                model.graph.remove_edge(i*n+j, (i+1)*n+j-1, weight = weight)"""
            if(j == n-1):
                if(i == n-1):
                    model.graph.remove_edge(i*n+j, 0)
                else:
                    model.graph.remove_edge(i*n+j, (i+1)*n)
                if(i == 0):
                    model.graph.remove_edge(i*n+j, (n-1)*n)
                else:
                    model.graph.remove_edge(i*n+j, (i-1)*n)
            if( i == n-1):
                if( j != n-1):
                    model.graph.remove_edge(i*n+j, j+1)
                if(j != 0):
                    model.graph.remove_edge(i*n+j, j-1)
                else: 
                    model.graph.remove_edge(i*n+j, (n-1))

def draw_model(model, save=True, filenumber = 1, outline=None, partition=None, extraTitle="plotting"):
    
    #plt.figure(figsize=(4, 4))
    #plt.subplot(1, 2, 1, title="State of the Nodes")
    color_map = []
    intensities = []
    #pos = []
    for node in model.graph:
        #pos.append(model.graph.nodes[node]['pos'])
        if model.graph.nodes[node]['agent'].state > 0:
            color_map.append((3/255,164/255,94/255, model.graph.nodes[node]['agent'].state))
            intensities.append(model.graph.nodes[node]['agent'].state)
            #color_map.append('#03a45e')
            #else: color_map.append('#f7796d')
            
        else: 
            color_map.append((247/255,121/255,109/255, -1*model.graph.nodes[node]['agent'].state ))
            intensities.append(model.graph.nodes[node]['agent'].state)
    degrees = nx.degree(model.graph)
    #plt.subplot(121)i
    nx.draw(model.graph, model.pos, node_size=[d[1] * 30 for d in degrees], linewidths=2, node_color =intensities, cmap=plt.cm.RdYlGn,  vmin=-1, vmax=1)  # REMOVE EDGELIST for auto drawing edges
    #nx.draw(model.graph, model.pos, node_size=[d[1] * 30 for d in degrees], linewidths=2, node_color =intensities, cmap=plt.cm.RdYlGn,  vmin=-1, vmax=1, edgelist = [])  # REMOVE EDGELIST for auto drawing edges
    #sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=-1, vmax=1))
    #sm.set_array([])
    #cbar = plt.colorbar(sm)
    #plt.colorbar(mcp)
    #plt.show()
    
    if(outline !=None):
        #mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "magenta","cyan", "cyan", "cyan", "cyan"]
        ax = plt.gca()
        ax.collections[0].set_edgecolor(outline)
        (clusters, sd, clsize) = findAvgStateInClusters(model, part= partition)
        text = [f'x={clusters[c]:5.2f} sd={sd[c]:5.2f} n={clsize[c]}' for c in range(len(clusters))]
        #print(text)
        handles = [mpatches.Patch(color=mypalette[c], label=text[c]) for c in range(len(text))]
        ax.legend(handles=handles)
        plt.title("Snapshot of network with states and clusters")


    if(save):
        plt.title(str(filenumber)+extraTitle)
        plt.savefig("plot" + str(filenumber) +".png", bbox_inches="tight")
        plt.close('all')
        

def drawAvgState(models, avg =False, pltNr=1, title="", clusterSD = False):
    plt.xlabel("timestep")
    plt.ylabel("AVG // STD")
    #mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "grey","grey"]
    plt.subplot()
    #plt.subplot(1, 2, 1, title="Average State and SD")
    
    if(not avg):
        plt.ylim((-1, 1))
        for i in range(len(models)):
            plt.plot(models[i].states ,color='#ff7f0e')
            plt.plot(models[i].statesds, alpha=0.5 ,color='#ff7f0e')
            if(clusterSD):
                sds = np.array(models[i].clusterSD)
                avgsd = sds.mean(axis=1)
                plt.plot(avgsd, linestyle=":" ,color='#ff7f0e')
    else:
        states = []
        sds = []
        plt.ylim((-1, 1))
        for i in range(len(models)):
            states.append(models[i].states)
            sds.append(models[i].statesds)
        array = np.array(states)
        avg = array.mean(axis=0)
        std = np.array(sds).mean(axis=0)
        p1 = plt.plot(avg, color='#ff7f0e', label="AVG state")
        p2 = plt.plot(std, color='#ff7f0e', alpha=0.5, label="STD states")
        #plt.plot(avg+std, color=col.to_rgba(mypalette[pltNr-1], 0.5))
        #text = ["rand cont", "cl cont", "rand disc", "cl disc"]
        text =["Clustered"]
        handles = [mpatches.Patch(color=mypalette[c], label=text[c]) for c in range(len(text))]
        plt.legend(handles=handles)
        #print(models[0].states)
        if(clusterSD):
            avgSds = []
            for mod in models:
                array = np.array(mod.clusterSD)
                avgSd = array.mean(axis=1)
                avgSds.append(avgSd)
            array = np.array(avgSds)
            avgAvgSd = array.mean(axis=0)
            plt.plot(avgAvgSd, color='#ff7f0e', linestyle=":", label="STD in clusters")

        #plt.subplot(1, 2, 2)
        #plt.ylim((0, 1))
        #plt.plot(std, color=mypalette[pltNr-1])
        return (p1, p2)

def drawCrossSection(models, pltNr = 1):
    values = []
    #mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "grey","grey"]
    for model in models:
        values.append(model.states[-1])
    plt.subplot(1, 2, 2, title="Density Plot of State for Simulations")
    ax = plt.gca()
    #ax.set_xscale('log')
    plt.xlim((0, 5))
    plt.ylim((-1, 1))
    #plt.title('Density Plot of state for simulations')
    #plt.xlabel('avg state of cooperators after all time steps')
    plt.xlabel('Density')
    #plt.scatter(range(len(values)), values.sort(), color = mypalette[pltNr-1])
    try:
        sns.distplot(values, hist=False, kde=True, color = mypalette[pltNr-1], vertical=True)
    except:
        sns.distplot(values, hist=True, kde=False, color = mypalette[pltNr-1], vertical=True)

    #plt.show()

def drawClustersizes(models, pltNr = 1):
    sizes = []
    for model in models:
        part = findClusters(model)
        (avg, sd, size) = findAvgStateInClusters(model, part)
        for s in size:
            sizes.append(s)
    plt.subplot(1, 3, 3, title="Density Plot of clustersize simulations")
    plt.xlabel("Clustersize")
    sns.distplot(sizes, hist=True, kde=True, color = mypalette[pltNr-1])

def drawConvergence(variables, modelsList, pltNr = 1):
    endState = []
    for models in modelsList:
        values = []
        for model in models:
            values.append(model.states[-1])
        endState.append(mean(values))
    plt.subplot(1,2,2)
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.scatter(variables, endState, color=mypalette[pltNr-1])

def drawClusterState(models, pltNr = 1, step=-1, subplot=1):
    plt.title("Density of Avg State in Communities")
    if(step < 0):
        plt.subplot(1, 3, 3, title="Avg State after Simulation")
        states = []
        for i in range(len(models)):
            for c in models[i].clusteravg[-1]:
                states.append(c)
    else:
        plt.subplot(1, 3, subplot, title="Avg State at t="+ str(step))
        states = []
        for i in range(len(models)):
            for c in models[i].clusteravg[step]:
                states.append(c)
    ax = plt.gca()
    #ax.set_xscale('log')
    plt.xlim((0, 5))
    plt.ylim((-1, 1))
    #plt.title('Density Plot of state for simulations')
    #plt.xlabel('avg state of cooperators after all time steps')
    plt.xlabel('Density')
    plt.ylabel('State')
    try:
        sns.distplot(states, hist=True, kde=True, color = mypalette[pltNr-1], vertical=True)
    except:
        sns.distplot(states, hist=True, kde=False, color = mypalette[pltNr-1], vertical=True)

def drawAvgNumberOfAgreeingFriends(models, pltNr = 1):
    avgNbAgreeingFriends = [model.avgNbAgreeingList for model in models]
    avgAvg = np.array(avgNbAgreeingFriends).mean(axis=0)
    plt.title("Average Agreement of Neighbours")
    plt.ylim((0, 1))
    plt.xlabel("Timesteps")
    plt.ylabel("Agreement")
    plt.plot(avgAvg, color=mypalette[pltNr-1])
