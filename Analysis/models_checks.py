#To Check:
#link is broken only if link established
#check why average degree of bridge_if_same is declining 
# in bridge and biased rewiring agents establish links before breaking, random is opposite
#switch code so 



#%%
import os 
import sys
# Ensure the parent directory is in the sys.path for auxiliary imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

#%%
import numpy as np
import random 
import threading
#sys.path.append("..")
import pandas as pd
from copy import deepcopy
from statistics import stdev, mean
import imageio
import networkx as nx
from networkx.algorithms.community import louvain_communities as community_louvain
from scipy.stats import truncnorm
from operator import itemgetter
import heapq
from IPython.display import Image
import time
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pickle
import igraph as ig
import leidenalg as la
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from netin import DPAH, PATCH, viz, stats
from netin.generators.h import Homophily
from collections import Counter
from Auxillary import network_stats
import rustworkx as rx
import multiprocessing
import subprocess
from Auxillary import node2vec_cpp as n2v

#%%

#random.seed(1574705741)    ## if you need to set a specific random seed put it here
#np.random.seed(1574705741)
#Helper functions

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def setArgs(newArgs):
    global args
    for arg, value in newArgs.items():
        args[arg] = value

def getRandomExpo():
    x = np.random.exponential(scale=0.6667)-1
    if(x>1): return 1
    elif (x< -1): return -1
    return x

def update_instance_methods(instance, func_changes):
    for func in func_changes:
        # Bind the method to the instance
        bound_method = func.__get__(instance, instance.__class__)
        instance.call_algo = bound_method
        setattr(instance, func.__name__, bound_method)
#Constants and Variables

## for explanation of these, see David's paper / thesis (Dynamics of collective action to conserve a large common-pool resource // Simple models for complex nonequilibrium Problems in nanoscale friction and network dynamics)
STATES = [1, -1] #1 being cooperating, -1 being defecting
defectorUtility = 0.0 # not used anymore
politicalClimate = 0.05 
newPoliticalClimate = 1*politicalClimate # we can change the political climate mid run
stubbornness = 0.6
degree = 8 
timesteps= 100 #70000 
continuous = True
skew = -0.25
initSD = 0.15
mypalette = ["blue","red","green", "orange", "magenta","cyan","violet", "grey", "yellow"] # set the colot of plots
randomness = 0.10
gridtype = 'cl' # this is actually set in run.py for some reason... overrides this
gridsize = 33   # used for grid networks
nwsize = 100 #1089  # nwsize = 1089 used for CSF (Clustered scale free network) networks
friendship = 0.5
friendshipSD = 0.15
clustering = 0.5 # CSF clustering in louvain algorithm
#new variables:
breaklinkprob = 0.5
rewiringMode = "None"
polarisingNode_f = 0.10
establishlinkprob = 0.5 # breaklinkprob and establishlinkprob are used in random rewiring. Are always chosen to be the same to keep average degree constant!
rewiringAlgorithm = 'None' #None, random, biased, bridge
#the rewiringAlgorithm variable was meant to enable to do multiple runs at once. However the loop where the specification 
#given in the ArgList in run.py file overrules what is given in line 65 does not work. Unclear why. 
#long story short. All changes to breaklinkprob, establishlinkprob and rewiringAlgorithm have to be specified here in the models file
lock = None

#print(os.getcwd())
# the arguments provided in run.py overrides these values
args = {"defectorUtility" : defectorUtility, 
        "politicalClimate" : politicalClimate, 
        "stubbornness": stubbornness, "degree":degree, "timesteps" : timesteps, "continuous" : continuous, "type" : gridtype, "skew": skew, "initSD": initSD, "newPoliticalClimate": newPoliticalClimate, "randomness" : randomness, "friendship" : friendship, "friendshipSD" : friendshipSD, "clustering" : clustering,
        "rewiringAlgorithm" : rewiringAlgorithm,
        "plot": False,
        "nwsize": nwsize,
        "rewiringMode": rewiringMode,
        "breaklinkprob" : breaklinkprob,
        "establishlinkprob" : establishlinkprob,
        "polarisingNode_f": polarisingNode_f}

#%% simulate 
def getargs():
    return args

def init_lock(lock_):
    global lock
    lock = lock_

def simulate(i, newArgs, func_changes = False): #RG for random graph (used for testing)
    setArgs(newArgs)
    #global args

    # random number generators
    friendshipWeightGenerator = get_truncated_normal(args["friendship"], args["friendshipSD"], 0, 1) 
    initialStateGenerator = get_truncated_normal(args["skew"], args["initSD"], -1, 1)


    # network type to use, I always ran on a cl 
    if(args["type"] == "cl"):
        model =ClusteredPowerlawModel(args["nwsize"], args["degree"], skew=args["skew"], friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
  
    elif(args["type"] == "sf"):
        model = ScaleFreeModel(args["nwsize"], args["degree"], skew=args["skew"], friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
    elif(args["type"] == "rand"):
        model = RandomModel(args["nwsize"], args["degree"], skew=args["skew"], friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
    elif(args["type"] == "FB"):
        model = EmpiricalModel(f"../Pre_processing/networks_processed/{args['top_file']}", args["nwsize"], args["degree"],  skew=args["skew"], friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
    elif(args["type"] == "Twitter"):
        model = EmpiricalModel(f"../Pre_processing/networks_processed/{args['top_file']}", args["nwsize"], args["degree"],  skew=args["skew"], friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
    elif(args["type"] == "DPAH"):
        #args degre a.k.a 'm' is just passed here as a dummy variable but does not affect the DPAH model
        model = DPAHModel(args["nwsize"], args["degree"], skew=args["skew"], friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
    else:
        model = RandomModel(args["nwsize"], args["degree"],  friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
    
    if func_changes:
        update_instance_methods(model, func_changes)
    
    #model.lock = lock
    res = model.runSim(args["timesteps"], clusters=True, drawModel=args["plot"]) ## gifname provide a gif name if you want a gif animation, need to specify time stamps later on
    #C_end, S_end = model.property_checks(R_G)

    #draw_model(model)
    return model


#%% Agent class 
class Agent:
    def __init__(self, state, stubbornness):
        self.state = state 
        self.interactionsReceived = 0
        self.interactionsGiven = 0
        self.stubbornness = stubbornness
        self.type = "converging"

    # this part is important, it defines the agent-agent interaction
    def consider(self, neighbour, neighboursWeight, politicalClimate):
        self.interactionsReceived +=1
        neighbour.addInteractionGiven()
        if(self.stubbornness >= 1): return
        
        mod = 1 
        if (self.type == "polarising") & (self.state*neighbour.state < 0):
            mod = -1 
     
        global args
        weight = self.state*self.stubbornness + politicalClimate + args["defectorUtility"] + neighboursWeight*neighbour.state 


        p1 = 0
        sample = random.uniform(-randomness,randomness)
        check = (weight + sample)

        if(check < -randomness): 
            p1 = 0

        elif(check > randomness): 
            p1 = 1

        else: 
            p1 = 1/(2*randomness)*(randomness + sample)

        p2 = 1 - p1
        delta = mod*(abs(self.state - neighbour.state)*(p1*(1-self.state) - p2*(1+self.state)))

        self.state += delta

        if(self.state > 1):
            self.state = STATES[0]
        elif(self.state <-1):
            self.state = STATES[1]       

    def addInteractionGiven(self):
        self.interactionsGiven +=1

    def setState(self, newState):
        if(newState >= STATES[1] and newState <= STATES[0]):
            self.state = newState
        else:
            print("Error state outside state range: ", newState)

#%% Model 

# this class contains functions and values partaining to the model, some of it is in use, some of it is not
class Model:
    # initial values
    def __init__(self, friendshipWeightGenerator = None, initialStateGenerator=None):
        global args
        self.graph = nx.Graph()
        self.politicalClimate = args["politicalClimate"]
        self.ratio = []
        self.states = []
        self.statesds = []
        self.clustering = []
        self.degrees = []
        self.degreesSD = []
        self.mindegrees_l = []
        self.maxdegrees_l = []
        self.defectorDefectingNeighsList = []
        self.cooperatorDefectingNeighsList = []
        self.defectorDefectingNeighsSTDList = []
        self.cooperatorDefectingNeighsSTDList =[]
        self.pos = []
        self.friendshipWeightGenerator = friendshipWeightGenerator
        self.initialStateGenerator = initialStateGenerator
        self.clusteravg = []
        self.clusterSD = []
        self.NbAgreeingFriends = []
        self.avgNbAgreeingList = []
        self.partition = None
        self.test = []
        self.clustering_diff = []
        self.small_world_diff = []
        self.node2vec_executable = n2v.get_node2vec_path()
        self.affected_nodes = []
        self.embeddings = {}
        self.retrain = 0
        self.process_id = multiprocessing.current_process().pid
 
         
    
        #setting rewiring algorithm to be used
        rewiringAlgorithm = args["rewiringAlgorithm"]
        if rewiringAlgorithm == "None": 
            rewiringAlgorithm = None
        #print(rewiringAlgorithm)
        self.algo = rewiringAlgorithm
        #the same random agent rewires after interacting:
        #TODO only need to call this once, need to change
        if rewiringAlgorithm != None:
            self.interact_main = self.interact
            
            if rewiringAlgorithm == 'random':
               self.call_algo = self.randomrewiring
            elif rewiringAlgorithm == 'biased':
                self.call_algo = self.biasedrewiring
            elif rewiringAlgorithm == 'bridge':
                self.call_algo = self.bridgerewiring
                
            elif rewiringAlgorithm == "wtf":
                    
                self.call_algo = self.call_wtf
                
            elif rewiringAlgorithm == "node2vec":
             
                self.call_algo = self.call_node2vec
        #otherwise we are using the static scenario and we don't include rewiring
        else:
            self.interact_main = self.interact_init

         
    # picks a randon agent to perform an interaction with a random neighbour and then to rewire
    def interact(self):
        #print('starting interaction')
        nodeIndex = random.randint(0, len(self.graph) - 1)
        #print("in interact: ", nodeIndex)
        node = self.graph.nodes[nodeIndex]['agent']
     
            
        neighbours =  list(self.graph.adj[nodeIndex].keys())
        
        if(len(neighbours) == 0):
            return nodeIndex
       
        
        chosenNeighbourIndex = neighbours[random.randint(0, len(neighbours)-1)]
        chosenNeighbour = self.graph.nodes[chosenNeighbourIndex]['agent']
        weight = self.graph[nodeIndex][chosenNeighbourIndex]['weight']

        node.consider(chosenNeighbour, weight, self.politicalClimate)
        
        self.call_algo(nodeIndex)
    
    
    #static interaction for generating networks and static mode
    def interact_init(self):
        #print('starting interaction')
        nodeIndex = random.randint(0, len(self.graph) - 1)
        #print("in interact: ", nodeIndex)
        node = self.graph.nodes[nodeIndex]['agent']
     
            
        neighbours =  list(self.graph.adj[nodeIndex].keys())
        if(len(neighbours) == 0):
            return nodeIndex
        
        chosenNeighbourIndex = neighbours[random.randint(0, len(neighbours)-1)]
        chosenNeighbour = self.graph.nodes[chosenNeighbourIndex]['agent']
        weight = self.graph[nodeIndex][chosenNeighbourIndex]['weight']

        node.consider(chosenNeighbour, weight, self.politicalClimate)
        
        #print('ending interaction')
        return nodeIndex
    
   

#%%% Rewiring algorithms
#rewiring--------------------------------------------------------------------------------
    
    def quick_sigma(self, test_graph, R_G, *args):
            
        def calc(graph):
            C, L = nx.average_clustering(graph), nx.average_shortest_path_length(graph)
            return C, L 
            
        out  = list(map(calc, [R_G, test_graph]))
        
        Cr, Lr, C, L = list(sum(out,()))
        
        sw_sigma = (C/Cr)/(L/Lr)
                
        return sw_sigma  
    
    
    #selectes node with highest degree from node_list
    def select_kmax_new(self, node_list):
        
        degree_sum = 0
        degree_sum = sum(self.graph.degree(i) for i in node_list)
        
        node = None 
        while node == None:
            for i in node_list:
                prob = self.graph.degree(i)/degree_sum
                
                if random.random() < prob:
                    node = i 
                    break
            
        return node 
    

    def check_node_state(self, node, neighbor):
        
        node_state = ""
         
        if (node.state >=0 and neighbor.state <0) or (node.state <0 and neighbor.state >=0):
                node_state = "different"
        elif (node.state >=0 and neighbor.state >=0) or (node.state <0 and neighbor.state <0):
                node_state = "same"
                
        return node_state
    
    def rewire(self, node_i, neighbour_i):
        
        rewired = False
        if random.random() < establishlinkprob:
            weight = get_truncated_normal(args["friendship"], args["friendshipSD"], 0, 1).rvs(1)[0]
            
            self.graph.add_edge(node_i, neighbour_i, weight = weight)
            #self.TF_step(node_i, neighbour_i, weight = weight)
            rewired = True
            
        return rewired 
    

    #triad formation
    def TF_step(self, node, node_new, weight):
        
        adjacency = list(self.graph.adj[node_new].keys())
        
        if len(adjacency) > 1:
            #make sure node doesn't conect back to itself
            adjacency.remove(node)
            
            #print(adjacency)
            neighbour_node = random.sample(adjacency, 1)[0]
            
            #print(neighbour_node)
            self.graph.add_edge(node, neighbour_node, weight = weight)
         
            
        return 
    
        
        
    #this takes too long currently
    def property_checks(self, R_G): 
    
        
        #small world property "sigma"
        #sw_property = nx.sigma(self.graph)
        sw_property = self.quick_sigma(self.graph, R_G, args)
        #calculate avg clustering
        clustering = nx.average_clustering(self.graph)
        
        return clustering, sw_property
    
 
    def find_2_steps(self, nodeIndex):
        
        adjacency = list(self.graph.adj[nodeIndex].keys())
        total_neighbours = []
        
        for i in adjacency:
            adjacency_two = list(self.graph.adj[i].keys())
            total_neighbours+= adjacency_two
        
        #removing nodeIndex
        total_neighbours = [x for x in total_neighbours if x != nodeIndex]
        
        assert nodeIndex not in total_neighbours, "index in list"
        
        return total_neighbours 
    
    
    def randomrewiring(self, nodeIndex, establishlinkprob = args["establishlinkprob"]):
        
               
           # print('starting random rewiring')
            
            #reseting rewired value
            rewired_rand = False 
            
            #establishlinkprob = args["establishlinkprob"]
            breaklinkprob = args["breaklinkprob"]
            self.probs  = establishlinkprob, breaklinkprob
                
            non_neighbors = []
            non_neighbors.append([k for k in nx.non_neighbors(self.graph, nodeIndex)])
            non_neighbors = non_neighbors[0]
           
            if(len(non_neighbors) == 0): #the agent is connected to the whole network, hence cannot establish further links
                return nodeIndex
            else:
                establishlinkNeighborIndex = random.sample(non_neighbors, 1)
                if random.random() < establishlinkprob:
                   # print('establishing link')
                    self.graph.add_edge(nodeIndex, establishlinkNeighborIndex[0], weight = get_truncated_normal(args["friendship"], args["friendshipSD"], 0, 1).rvs(1)[0])
                    # print(self.graph.edges[nodeIndex, establishlinkNeighborIndex[0]]['weight'])
                    rewired_rand = True 
                    
            init_neighbours =  list(self.graph.adj[nodeIndex].keys())
            
            if rewired_rand == True:
                
                 if(len(init_neighbours) < 2):
                    return nodeIndex
            
                 else:
                    breaklinkNeighbourIndex = init_neighbours[random.randint(0, len(init_neighbours)-1)]
           
                    if random.random() < breaklinkprob:
                     #  print('breaking link')
                       self.graph.remove_edge(nodeIndex, breaklinkNeighbourIndex)                                
                
            
            # print('ending random rewiring')
            
        
        
    def biasedrewiring(self, nodeIndex):
        #'realistic' rewiring. Agent can only rewire within small number of network steps and has a bias towards like minded people in the establishing of links
        #here we have an extreme assumption: if agents are of the same opinion a link is established for sure, if not, no link is established -> could be refined by introducing a probability curve 
        #e.g the more alike they are, the higher the probability of establishing a link, vice versa
        #breaking a link only when a link is established is a good idea as the likelyhood of meeting like thinkers changes as the network evolves and it is an easy way to retain the average degree to not get isolated nodes etc
      
        #print('starting biased rewiring')        
        
        potential_friends = []
        non_neighbors = []
        non_neighbors.append([k for k in nx.non_neighbors(self.graph,nodeIndex)])
        non_neighbors = non_neighbors[0]
        #print("first neighbours: ", non_neighbors)
        
        #finding neighbors
        total_neighbours = self.find_2_steps(nodeIndex)
        #print(total_neighbours) 
        
        #removing duplicates
        res = [*set(total_neighbours)]
        
        #print(res)
        
        potential_friends = set(non_neighbors).intersection(set(res))
        

                   
        rewired = False #only if a link is established a link is broken hence we need a variable telling us whether a link has been established
        #print(rewired)
              
        if(len(potential_friends) == 0): #the agent is connected to the whole network, hence cannot establish further links
           return nodeIndex
        else:
           establishlinkNeighborIndex = self.select_kmax_new(potential_friends) #here preferential attachment should be implemented
           #print(establishlinkNeighborIndex)
               
           node = self.graph.nodes[nodeIndex]['agent']
           establishlinkNeighbor = self.graph.nodes[establishlinkNeighborIndex]['agent']
          
           node_state = self.check_node_state(node, establishlinkNeighbor)
           
           rewiring_mode = args["rewiringMode"]
           
           if node_state in "different" and rewiring_mode in "diff":
               rewired = self.rewire(nodeIndex, establishlinkNeighborIndex) 
              
                     
           elif node_state in "same" and rewiring_mode in "same":
               rewired = self.rewire(nodeIndex, establishlinkNeighborIndex) 
             
           else:
               return nodeIndex
               
                   
           if rewired == True: #links are only broken if a link is established
                
                init_neighbours =  list(self.graph.adj[nodeIndex].keys())
                if(len(init_neighbours) < 2):
                    return nodeIndex
            
                else:
                     #print('breaking a link')
                     breaklinkNeighbourIndex = init_neighbours[random.randint(0, len(init_neighbours)-1)]
                     self.graph.remove_edge(nodeIndex, breaklinkNeighbourIndex)


    def bridgerewiring(self, nodeIndex):
        #rewiring outside of ones own (opinion) cluster (we work with louvain clusters (=topological) but these become proxies for opinion clusters over time) -> time efficiency reason (see thesis)
        #we are looking at a deliberate algorithm that promotes faster dissolution of clusters (extreme case), non realistic
        #links are established for sure if agents disagree in their opinion, no link is established if they agree;
        #again this could be refined by introducing probability functions
        #links are only broken is a link is established (see above in biasedrewiring)
              
        if(self.partition == None): #louvain is already used in first step of runSim, this is just to set the variable equal to the partition in the function as well.
            partition = findClusters(self.graph, self.community_detection_with_leidenalg)
        else:
            partition = self.partition
            
        for k, v in partition.items():
            self.graph.nodes[k]["louvain-val"] = v

        
        nodeCluster =  self.graph.nodes[nodeIndex]['louvain-val'] #which cluster does agent i belong to?
        #print(nodeCluster)
        other_cluster = []
        other_cluster.append([k for k in self.graph.nodes if k != nodeIndex and self.graph.nodes[k]['louvain-val'] != nodeCluster]) #a list of all nodes that are not in agent i's cluster
        other_cluster = other_cluster[0]
        #print(other_cluster)
        rewired = False
        
        if(len(other_cluster) == 0): #the agent is connected to the whole network, hence cannot establish further links
            return nodeIndex
            #print('connected to everyone')
        else:
            #print('bridge rewiring')
            partnerOutClusterIndex = self.select_kmax_new(other_cluster) #here preferential attachment should be implemented
            #print(partnerOutClusterIndex)
            #print(self.graph.nodes[partnerOutClusterIndex]['louvain-val'])
                
            node = self.graph.nodes[nodeIndex]['agent']
            partnerOutCluster = self.graph.nodes[partnerOutClusterIndex]['agent']
            
            node_state = self.check_node_state(node, partnerOutCluster)
            
            rewiring_mode = args["rewiringMode"]
        
            
            if node_state in "different" and rewiring_mode in "diff":
                rewired = self.rewire(nodeIndex, partnerOutClusterIndex) 
                #self.TF_step(nodeIndex, partnerOutClusterIndex, weight)

    
                      
            elif node_state in "same" and rewiring_mode in "same":
                #returns true or false depending on success
                rewired = self.rewire(nodeIndex, partnerOutClusterIndex) 
                #self.TF_step(nodeIndex, partnerOutClusterIndex, weight)
                
    
              
            else:
                return nodeIndex
            
            if rewired == True:
                if random.random() < breaklinkprob:
                    init_neighbours =  list(self.graph.adj[nodeIndex].keys())
                    if(len(init_neighbours) < 2):
                        return nodeIndex
                     
                    else:
                        breaklinkNeighbourIndex = init_neighbours[random.randint(0, len(init_neighbours)-1)]
                            
                        self.graph.remove_edge(nodeIndex, breaklinkNeighbourIndex)
    #-----------------------------------------------------------------------------------
          
        return nodeIndex
    
    def break_link(self, nodeIndex, rewiredIndex, neighbours):
        
        #avoids repeated computation
        init_neighbours = neighbours
        
        #taking out index of freshly rewired node
        reduced = [x for x in init_neighbours if x != rewiredIndex]
        
        if(len(reduced) < 2): #random.random() >= breaklinkprob):
            return nodeIndex, False
         
        else:
            breaklinkNeighbourIndex = np.random.choice(reduced)
            #assert breaklinkNeighbourIndex != rewiredIndex, "they just became friends!"
            self.graph.remove_edge(nodeIndex, breaklinkNeighbourIndex)
            return breaklinkNeighbourIndex, True
                    

    def train_node2vec(self, input_file='graph.edgelist', output_file='embeddings.emb', dimensions=64):
       
        if not hasattr(self, 'affected_nodes_set'):
            self.affected_nodes_set = set()
        
        # Convert affected_nodes list to a set for efficient lookup
        self.affected_nodes_set.update(self.affected_nodes)
        
        # Use process-specific file names
        process_input_file = f"{input_file}_{self.process_id}"
        process_output_file = f"{output_file}_{self.process_id}"
    
        if not self.embeddings or len(self.affected_nodes_set) > len(self.graph.nodes) * 0.5:
            # If embeddings don't exist or too many nodes are affected, retrain all
            n2v.save_graph_as_edgelist(self.graph, process_input_file)
        else:
            # Partial retraining: save only edges connected to affected nodes
            with open(process_input_file, 'w') as f:
                for edge in self.graph.edges():
                    if edge[0] in self.affected_nodes or edge[1] in self.affected_nodes:
                        f.write(f"{edge[0]} {edge[1]}\n")
        
        # Run node2vec on the saved edgelist
        n2v.run_node2vec(self.node2vec_executable, process_input_file, process_output_file)
        new_embeddings = n2v.load_embeddings(process_output_file)
        
        # Update embeddings
        if not self.embeddings:
            self.embeddings = new_embeddings
        else:
            # Update only the affected nodes in the main embeddings
            for node in new_embeddings:
                self.embeddings[node] = new_embeddings[node]
        
        # Clear the affected nodes list after retraining
        self.affected_nodes_set.clear()
    
        # Clean up process-specific files
        os.remove(process_input_file)
  
    
    
    
           

    def node2vec_rewire(self, nodeIndex):
        
    
        def get_similar_agents(nodeIndex, embeddings=self.embeddings):
            
    
            target_vec = self.embeddings[nodeIndex]
            all_agents = list(self.embeddings.keys())
            all_vectors = np.array([self.embeddings[agent] for agent in all_agents])
 
            similarity = np.dot(all_vectors, target_vec) / (np.linalg.norm(all_vectors, axis=1) * np.linalg.norm(target_vec))
            similarities = [(agent, sim) for agent, sim in zip(all_agents, similarity) if agent != nodeIndex]
            similarities.sort(key=lambda x: x[1], reverse=True)
            #print(nodeIndex, similarities)
            return similarities
        
        #first index references matrix entry(tupple), second indexes first value of tupple(node Index of agent)
        similar_neighbours = np.array([x for x, y in get_similar_agents(nodeIndex)])
        sim = similar_neighbours[0]
    
        sim, neighbours = self.neighbours_check(nodeIndex, sim, similar_neighbours)
        
        #exits function if sim is None
        if sim is None:
            return False
            print('sim is None')
        
        #returns whether node was rewired
        rewire = self.rewire(nodeIndex, sim)
        
        if rewire:    
            break_link = self.break_link(nodeIndex, sim, neighbours)
            self.affected_nodes_set.update([nodeIndex, sim, break_link])

    
        return rewire
     
        
        
    
    
    def neighbours_check(self, nodeIndex, rewireIndex, potentialIndexes):
        
        
        neighbours = list(self.graph.adj[nodeIndex].keys())
        
        if len(neighbours) == self.graph.size():
          return None, neighbours
    
        #check that we are not rewiring to neighbours
        rank = 1
        while rewireIndex in neighbours:
            #print(rank)
            if rank <= len(neighbours)-1:
                #print(rank, neighbours)
                rewireIndex = potentialIndexes[rank]
                rank += 1
            else:
                return rewireIndex, neighbours
        return rewireIndex, neighbours
        
    def call_node2vec(self, nodeIndex):
        if not self.trained and nodeIndex in self.affected_nodes_set:    
            #self.test.append(self.t)
            self.train_node2vec()
            self.trained = True
    
        rewired = self.node2vec_rewire(nodeIndex)
           
        if rewired:
            self.trained = False
              
            
          

    def wtf_1(self, topk=5, alpha=0.70, max_iter=50):
        
        TOPK = topk

        # Convert NetworkX graph to rustworkx graph
        G = rx.networkx_converter(self.graph)
        
        def _ppr(nodeIndex, G, p, top, max_iter):
            pp = {node: 0 for node in G.nodes()}
            pp[nodeIndex] = 1.0
            pr = rx.pagerank(G, alpha=p, personalization=pp, max_iter=max_iter)
            pr_values = np.array(list(pr.values()))
            pr_indices = np.argsort(pr_values)[::-1]
            pr_indices = pr_indices[pr_indices != nodeIndex][:top]
            return pr_indices

        def get_circle_of_trust_per_node(G, p, top, max_iter):
            return [_ppr(nodeIndex, G, p, top, max_iter) for nodeIndex in range(len(G.nodes()))]

        def frequency_by_circle_of_trust(G, cot_per_node, top):
            unique_elements, counts_elements = np.unique(np.concatenate(cot_per_node), return_counts=True)
            count_dict = {el: count for el, count in zip(unique_elements, counts_elements)}
            return [count_dict.get(nodeIndex, 0) for nodeIndex in range(len(G.nodes()))]

        def _salsa(nodeIndex, cot, G, top):
            BG = rx.PyGraph()
            hubs = [f'h{vi}' for vi in cot]
            hub_indices = BG.add_nodes_from(hubs)
            edges = [(f'h{vi}', vj) for vi in cot for vj in G.neighbors(vi)]
            authorities = list(set(e[1] for e in edges))
            auth_indices = BG.add_nodes_from(authorities)
            
            hub_index_map = {h: idx for idx, h in enumerate(hubs)}
            auth_index_map = {a: idx for idx, a in enumerate(authorities)}
            
            edges = [(hub_index_map[f'h{vi}'], auth_index_map[vj]) for vi, vj in edges if f'h{vi}' in hub_index_map and vj in auth_index_map]
            
            BG.add_edges_from(edges)
            centrality = rx.eigenvector_centrality(BG)
            centrality = {n: c for n, c in centrality.items() if isinstance(n, int) and n not in cot and n != nodeIndex and n not in G.neighbors(nodeIndex)}
            sorted_centrality = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
            return np.array([n for n, _ in sorted_centrality[:top]])

        def frequency_by_who_to_follow(G, cot_per_node, top):
            results = [_salsa(nodeIndex, cot, G, top) for nodeIndex, cot in enumerate(cot_per_node)]
            unique_elements, counts_elements = np.unique(np.concatenate(results), return_counts=True)
            count_dict = {el: count for el, count in zip(unique_elements, counts_elements)}
            return [count_dict.get(nodeIndex, 0) for nodeIndex in range(len(G.nodes()))]

        def wtf_full_old(G, alpha, top, max_iter):
            cot_per_node = get_circle_of_trust_per_node(G, alpha, top, max_iter)
            wtf = frequency_by_who_to_follow(G, cot_per_node, top)
            return wtf

        # Calculate recommendations 
        self.ranking = wtf_full_old(G, alpha, TOPK, max_iter)
        
        return
      
 
    def wtf_rewire(self, nodeIndex):
        rewireIndex = np.argmax(self.ranking)
        rewireIndex, neighbours = self.neighbours_check(nodeIndex, rewireIndex, self.ranking)
        
        rewired = self.rewire(nodeIndex, rewireIndex)
        
        if rewired:    
            brokenIndex = self.break_link(nodeIndex, rewireIndex, neighbours)
            #need to check how brokenIndex works here
            self.affected_nodes += [nodeIndex, rewireIndex, brokenIndex]
            
        return rewired
            

        
    def call_wtf(self, nodeIndex):
        #checking if the ranking has been affected by rewiring previously
        if not self.trained and nodeIndex in self.affected_nodes:
            #self.retrain += 1 
            self.wtf_1()
            self.affected_nodes = []
            self.trained = True
        
        rewired = self.wtf_rewire(nodeIndex)
        
        if rewired:
            self.trained = False
    


    def getAvgState(self):
        states = []
        for node in self.graph:
            states.append(self.graph.nodes[node]['agent'].state)
        statearray = np.array(states)
        avg = statearray.mean(axis=0)
        sd = statearray.std()
        return (avg, sd)
    
    def getAvgDegree(self):
        degrees = [val for (node,val) in nx.degree(self.graph)]
        degreearray = np.array(degrees)
        mindegree = np.min(degreearray)
        maxdegree = np.max(degreearray)
        avgdegree = degreearray.mean(axis=0)
        sddegree = degreearray.std()
        return (avgdegree, sddegree, mindegree, maxdegree)
    
    def countCooperatorRatio(self):
        count = 0
        
        #print(list(self.graph.nodes))
        for node in self.graph.nodes:
            #print(node)
            if self.graph.nodes[node]['agent'].state > 0:
                count+=1
        return count/len(self.graph)

    
    def getFriendshipWeight(self):
        weigth = self.friendshipWeightGenerator.rvs(1)
        return weigth[0]

    #majority = 0, minority 1
    def getInitialState(self, node_state = False):
        global args
    
        state = self.initialStateGenerator.rvs(1)[0]
        
        #implicitly checks if node_state exists
        if node_state:
            while (node_state == 0 and state >= 0):
                state = self.initialStateGenerator.rvs(1)[0]
                #print("here")
        return state
    
#%% Model run

    # this part actually runs the simulation
    def runSim(self, steps, groupInteract=False, drawModel = False, countNeighbours = False, gifname= 'trialtrial', clusters= False):
      
        if(self.partition ==None):
            if nx.is_directed(self.graph):
                
                
                self.partition = self.community_detection_with_leidenalg(self.graph)
                
            else:
                
                self.partition = self.community_detection_with_leidenalg(self.graph)

           

        filenames = []

        if(countNeighbours):
            (defectorDefectingNeighs,
                    cooperatorDefectingFriends,
                    defectorDefectingNeighsSTD,
                    cooperatorDefectingFriendsSTD) = self.getAvgNumberOfDefectorNeigh()
            print("Defectors: avg: ", defectorDefectingNeighs, " std: ", defectorDefectingNeighsSTD)
            print("Cooperators: avg: ", cooperatorDefectingFriends, " std: ", cooperatorDefectingFriendsSTD)

        #create list of number of agreeing friends
        # self.findNbAgreeingFriends()
        # self.avgNbAgreeingList.append(mean(self.NbAgreeingFriends))
        
        global args 
        
        if args["rewiringAlgorithm"] in "node2vec":
            self.train_node2vec()
            self.trained = True
            #print("initial training complete")
            
        elif args["rewiringAlgorithm"] in "wtf":
            self.wtf_1()
            self.trained = True
            #print("initial training complete")
       
        for i in range(steps):
            
           # if 
        
            self.t = i 
        

            #print("step: ", i)
            nodeIndex = self.interact_main()
            ratio = self.countCooperatorRatio()
            self.ratio.append(ratio)
            (state, sd) = self.getAvgState()
            self.states.append(state)
            #self.clustering.append(nx.average_clustering(self.graph))
            self.statesds.append(sd)
            (degree, degreeSD, mindegree, maxdegree) = self.getAvgDegree()
            self.degrees.append(degree)
            self.degreesSD.append(degreeSD)
            self.mindegrees_l.append(mindegree)
            self.maxdegrees_l.append(maxdegree)
            # avgFriends = self.updateAvgNbAgreeingFriends(nodeIndex)
            #avgFriends = self.findNbAgreeingFriends(nodeIndex)
            #draw_model(self) #this should draw the model in every timestep! 
            # self.avgNbAgreeingList.append(avgFriends)
            
           
           
           # some book keeping
            if (clusters):
                (s, sds, size) = findAvgStateInClusters(self, self.partition)
                self.clusterSD.append(sds)
                self.clusteravg.append(s)

            if(countNeighbours):
                (defectorDefectingNeighs,
                        cooperatorDefectingNeighs,
                        defectorDefectingNeighsSTD,
                        cooperatorDefectingNeighsSTD) = self.getAvgNumberOfDefectorNeigh()
                self.defectorDefectingNeighsList.append(defectorDefectingNeighs)
                self.cooperatorDefectingNeighsList.append(cooperatorDefectingNeighs)
                self.defectorDefectingNeighsSTDList.append(defectorDefectingNeighsSTD)
                self.cooperatorDefectingNeighsSTDList.append(cooperatorDefectingNeighsSTD)
            
            snapshots = [0, int(args["timesteps"]/2), args["timesteps"]-1]
           
            if i in snapshots and drawModel:
                self.plot_network(self.graph, title = f"T = {i}, N = {args['nwsize']}")
            
                
            # if(gifname != None and (i in snapshots)):
            #     draw_model(self, True, i, extraTitle = f'  avg state: {self.states[-1]:1.2f} agreement: {self.avgNbAgreeingList[-1]:1.2f}')
            #     filenames.append("plot" + str(i) +".png")
            
        #if(gifname != None):
        #    images = []
        #    for filename in filenames:
        #        images.append(imageio.imread(filename))
        #    #0.08167
        #    imageio.mimsave("network" +gifname+ ".gif", images, duration=0.08167)

        (avgs, sds, sizes) = findAvgStateInClusters(self, self.partition)
        self.clusteravg.append(avgs)

        return self.ratio

    # initial generation of agents in network
    def populateModel(self, n, skew = 0):
        global args
        for i in range (n):
            
            agent1 = Agent(self.getInitialState(), args["stubbornness"])
            self.graph.nodes[i]['agent'] = agent1
            
            if args["polarisingNode_f"] > np.random.random():
                self.graph.nodes[i]['agent'].type = "polarising"
                
            # making sure no loners
            neighbours =  list(self.graph.adj[i].keys())
            
            if(len(neighbours) == 0):    
                self.randomrewiring(i, establishlinkprob = 1)
                
        edges = self.graph.edges() 
        for e in edges: 
            weight=self.getFriendshipWeight()
            self.graph[e[0]][e[1]]['weight'] = weight
            
    # this runs the model without rewiring for a while to align the opinion clusters
    #with the topological clusters
    
    def populateModel_empirical(self, n, skew=0):
        self.fraction_m, self.fraction_M = [], []
    
        for i in range(n):
            agent = Agent(self.getInitialState(), args["stubbornness"])
            self.graph.nodes[i]['agent'] = agent
            self.graph.nodes[i]["m"] = 1 if agent.state >= 0 else 0
    
            if args["polarisingNode_f"] > np.random.random():
                self.graph.nodes[i]['agent'].type = "polarising"
                
            neighbours =  list(self.graph.adj[i].keys())
            
            if(len(neighbours) == 0):    
                self.randomrewiring(i, establishlinkprob = 1)
        for e in self.graph.edges():
            self.graph[e[0]][e[1]]['weight'] = self.getFriendshipWeight()
        
        self.update_minority_fraction(n)
        minority_frac = sum(self.fraction_m) / n
        #print("before", minority_frac)
    
        h_m, h_M = network_stats.infer_homophily_values(self.graph, minority_frac)
        tolerance = 0.13
    
        while abs(h_m - h_M) > tolerance:
            self.interact_init()
    
            for i in range(n):
                self.graph.nodes[i]["m"] = 1 if self.graph.nodes[i]["agent"].state >= 0 else 0
    
            self.update_minority_fraction(n)
            minority_frac = sum(self.fraction_m) / n
            
            h_m, h_M = network_stats.infer_homophily_values(self.graph, minority_frac)
            #print(f"Homophily_delta: {abs(h_m - h_M)}, Homomphily(h_m, h_M): {h_m, h_M}")
        #print(f"Homophily_delta: {abs(h_m - h_M)}, Homomphily(h_m, h_M): {h_m, h_M}")
        

    def update_minority_fraction(self, n):
        
        self.fraction_m, self.fraction_M = [], []
        for i in range (n):
            
            node_class = self.graph.nodes[i]["m"]
            
            self.fraction_m.append(node_class) if node_class == 1 else self.fraction_M.append(node_class)
        
    
    def populateModel_netin(self, n, skew = 0):
        
        
        #for some reason the skew is offset by 0.01 after genreation, this brings it back to intended skew
        skew = skew- 0.01
        self.fraction_m, self.fraction_M = [], []
        for n in range (n):
            
            node_class = self.graph.nodes[n]["m"]
            
            neighbours =  list(self.graph.adj[n].keys())
            
            if(len(neighbours) == 0):    
                self.randomrewiring(n, establishlinkprob = 1)
            
            self.fraction_m.append(node_class) if node_class == 1 else self.fraction_M.append(node_class)
            
            agent1 = Agent(self.getInitialState(node_class), args["stubbornness"])
            
            self.graph.nodes[n]['agent'] = agent1
            
            if args["polarisingNode_f"] > np.random.random():
                self.graph.nodes[n]['agent'].type = "polarising"

        
        
        #print(Homophily.infer_homophily_values(self.graph))
        edges = self.graph.edges() 
        for e in edges: 
            weight=self.getFriendshipWeight()
            self.graph[e[0]][e[1]]['weight'] = weight
    

        
            
    def community_detection_with_leidenalg(self, nx_graph):
        # Convert networkx graph to igraph
        ig_graph = ig.Graph.from_networkx(nx_graph)
        
        # Perform community detection using leidenalg on the igraph graph
        partition = la.find_partition(ig_graph, la.ModularityVertexPartition)
        
        # Map the community detection results back to the networkx graph
        # Creating a dictionary where keys are nodes in the original networkx graph,
        # and values are their community labels
        nx_partition = {}
        for node in nx_graph.nodes():
            #ig_index = ig_graph.vs.find(name=str(node)).index  # Find the igraph index of the node
            nx_partition[node] = partition.membership[node]
        
        return nx_partition
    
    
    
    def plot_network(self, graph, colormap='coolwarm', title = False):
        """
        Plots a network with nodes colored according to their opinion values.
        
        Parameters:
        graph (networkx.Graph): The networkx graph with node attributes containing Agent objects.
        colormap (str): The name of the colormap to use for node coloring.
        """
        
        # Adjust spring layout parameters
        layout = nx.spring_layout(graph, k=0.3, iterations=50)
        labels = nx.get_node_attributes(graph, "m")
        
        # Extracting the opinions
        opinions = nx.get_node_attributes(graph, "agent")
        opinions = {k: v.state for k, v in opinions.items()}
        
        # Normalize the opinions to the range [-1, 1] for colormap
        norm = Normalize(vmin=-1, vmax=1)
        
        cmap = plt.cm.get_cmap(colormap).reversed()
        colors = [cmap(norm(opinions[node])) for node in graph.nodes]
    
        
        # Create a colormap scalar mappable for the colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Draw the graph
        plt.figure(figsize=(10, 7))
        ax = plt.gca()
        if args["type"] in ["FB", "Twitter"]:
            nx.draw(graph, labels=labels, arrows=nx.is_directed(graph), 
                    node_color=colors, with_labels=False, edge_color='gray', edgecolors = "black", node_size=190, 
                    font_size=10, alpha=0.9, ax=ax)
        else:  
            nx.draw(graph, pos=layout, labels=labels, arrows=nx.is_directed(graph), 
                    node_color=colors, with_labels=False, edge_color='gray', edgecolors = "black", node_size=190, 
                    font_size=10, alpha=0.9, ax=ax)
            
        # Adding a colorbar
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Opinion Value')
        
        # Add black border around the plot
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        
        # Show plot
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f'../Figs/Networks/graph_{title}_{args["type"]}_{args["rewiringAlgorithm"]}_{args["rewiringAlgorithm"]}.png', bbox_inches='tight', dpi = 300)
        plt.show()
        
#%% Network topologies 

class EmpiricalModel(Model):
    def __init__(self,  filename, n, m, skew= 0, **kwargs):
        super().__init__(**kwargs)
        with open(filename, 'rb') as f:
            self.graph = pickle.load(f)
      
        #nx.draw(self.graph, node_size = 12)
       # np.savetxt("debug.txt", list(self.graph.nodes))
      
        self.populateModel_empirical(n, skew)
        
class EmpiricalModel_w_agents(Model):
    def __init__(self,  filename, n, m, skew= 0, **kwargs):
        super().__init__(**kwargs)
        with open(filename, 'rb') as f:
            self.graph = pickle.load(f)
       # np.savetxt("debug.txt", list(self.graph.nodes))
        #self.populateModel_empirical(n, skew)
        #TODO: implement populate function for this model class
class DPAHModel(Model):
    def __init__(self, n, m, skew= 0, **kwargs):
        super().__init__(**kwargs)
        #TODO: make these not hard-coded 
        self.graph = DPAH(n, f_m=0.5, d=0.02, h_MM=0.75, h_mm=0.75, plo_M=2.0, plo_m=2.0,
                     seed = 42)
        
        self.graph.generate()
        
        self.populateModel_netin(n, skew)

        
class ScaleFreeModel(Model):
    def __init__(self, n, m, skew= 0, **kwargs):
        super().__init__(**kwargs)
        self.graph = nx.barabasi_albert_graph(n, m)
        self.populateModel(n, skew)

class ClusteredPowerlawModel(Model):
    def __init__(self, n, m, skew = 0, **kwargs):
        super().__init__(**kwargs)

        self.graph = PATCH(n =n, k = m, f_m=0.5, h_MM=0.75, h_mm=0.75, tc = clustering,
                     seed = 42)
        self.graph.generate()
        #print(Homophily.infer_homophily_values(self.graph))
        
        
        #self.graph = nx.powerlaw_cluster_graph(n, m, clustering)
        #self.populateModel(n, skew)
        self.populateModel_netin(n, skew)

class RandomModel(Model):
    def __init__(self, n, m, skew= 0, **kwargs):
        #m is avg degree/2
        super().__init__(**kwargs)
        p = 2*m/(n-1)

        self.graph =nx.erdos_renyi_graph(n, p)
        self.populateModel(n, skew)


def findClusters(G, algo = None):
    
    if nx.is_directed(G):
        
        partition = algo(G)
        
    else:
        
        partition = algo(G)
        
    #print(partition)
    return partition

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

    
#%% Auxillary and data collection functions
  
#-------- save data functions ---------

def saveavgdata(models, filename, args):
    # Get the maximum number of timesteps
    max_timesteps = max(len(model.states) for model in models)
    
    # Initialize arrays for storing data from all models
    all_states = np.zeros((len(models), max_timesteps))
    all_statesds = np.zeros((len(models), max_timesteps))
    all_degrees = np.zeros((len(models), max_timesteps))
    all_mindegrees = np.zeros((len(models), max_timesteps))
    all_maxdegrees = np.zeros((len(models), max_timesteps))
    
    all_individual_data = []
    
    for i, model in enumerate(models):
        timesteps = len(model.states)
        
        # Store data for each model
        all_states[i, :timesteps] = model.states
        all_statesds[i, :timesteps] = model.statesds
        all_degrees[i, :timesteps] = np.mean(model.degrees, axis=0)
        all_mindegrees[i, :timesteps] = np.mean(model.mindegrees_l, axis=0)
        all_maxdegrees[i, :timesteps] = np.mean(model.maxdegrees_l, axis=0)
        
        # Create individual model data
        individual_model_data = pd.DataFrame({
            't': np.arange(timesteps),
            'avg_state': model.states,
            'std_states': model.statesds,
            'model_run': i,
            'scenario': args["rewiringAlgorithm"],
            'rewiring': args["rewiringMode"],
            'type': args["type"]
        })
        
        all_individual_data.append(individual_model_data)
    
    # Calculate averages using NumPy functions
    avg_states = np.nanmean(all_states, axis=0)
    avg_statesds = np.nanmean(all_statesds, axis=0)
    avg_degrees = np.nanmean(all_degrees, axis=0)
    degree_sd = np.nanstd(all_degrees, axis=0)
    avg_mindegrees = np.nanmean(all_mindegrees, axis=0)
    avg_maxdegrees = np.nanmean(all_maxdegrees, axis=0)
    
    # Create a dictionary for the averaged data across all models
    avg_model_data = {
        't': np.arange(max_timesteps),
        'avg_state': avg_states,
        'std_states': avg_statesds,
        'avgdegree': avg_degrees,
        'degreeSD': degree_sd,
        'mindegree': avg_mindegrees,
        'maxdegree': avg_maxdegrees,
        'scenario': args["rewiringAlgorithm"],
        'rewiring': args["rewiringMode"],
        'type': args["type"]
    }
    
    # Create DataFrames
    combined_avg_df = pd.DataFrame(avg_model_data)
    combined_individual_df = pd.concat(all_individual_data, ignore_index=True)
    
    return combined_avg_df, combined_individual_df

def savesubdata(models,filename):
    
    outs = []

    for i in range(len(models)):
        outs.append(models[i].states)
    
    outs = np.array(outs)
    np.savetxt(filename,outs,delimiter=',')

#-------- drawing functions ---------

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





#%%
#for testing only
# from netin import stats

# def timer(func, arg):
#     start = time.time()
#     func(arg)
#     end = time.time()
#     mins = (end - start) / 60
#     sec = (end - start) % 60
#     print(f'Runtime was complete: {mins:5.0f} mins {sec}s\n')

#     return
def test_run():
    twitter, fb  = "twitter_graph_N_789", "FB_graph_N_786"
    init_states = []
    final_states = []
    start = time.time()
    plt.figure()
    model_array = []
    for i in range(2):
        print(i)
        args.update({"type": "DPAH", "plot": False, "top_file": f"{twitter}.gpickle", "timesteps": 1000, "rewiringAlgorithm": "node2vec",
                      "rewiringMode": "None", "nwsize":100})
        #nwsize has to equal empirical network size 
        model = simulate(1, args)
        init_states.append(model.states[0])
        states = model.states
        final_states.append(states[-1])
        plt.plot(states)
        model_array.append(model)
    
        
    plt.ylim(-1, 1)
    plt.title(f'{args["rewiringAlgorithm"]}')
    plt.axline((0, np.mean(final_states)), slope= 0, color ="black")
    plt.show()  # Ensure plot is rendered
    return model_array
    
if  __name__ ==  '__main__': 
    start = time.time()
    models = test_run()
    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Runtime was complete: {mins:5.0f} mins {sec}s\n')


    fname = f'../Output/lol.csv'
    out = saveavgdata(models, fname, args)
#baseline run node2vec is 1 min 7 
    
# plt.savefig(f'../Figs/_{args["rewiringAlgorithm"]}_full_args_{args["nwsize"]}_{args["timesteps"]}.jpg')
# print(np.mean([models[i].retrain for i in range(len(models))]))
# print(np.std([models[i].retrain for i in range(len(models))]))
#227, 3.38 for last step rewire


# # final_states_compile_list = list(zip(final_states_2nd, final_states_3rd))
# # final_states_compiled = pd.DataFrame(final_states_compile_list, columns = ["final_states_t_2", 'final_states_full'])
# # final_states_compiled.to_csv("final_states_node2vec_test.csv")
# #final_states_2nd = final_states.copy()
# end = time.time()
# mins = (end - start) / 60
# sec = (end - start) % 60
# print(f'Runtime was complete: {mins:5.0f} mins {sec}s\n')

# states = model.states
# plt.plot(states)
# plt.show
# # plt.plot(init_states)
# # plt.axhline(y=np.mean(init_states), color='r', linestyle='-')
# # plt.show()
# # #viz.plot_graph(model.graph, edge_width = 1, cell_size = 3, node_size = 50)
# # # # hom = Homophily.infer_homophily_values(model.graph)



# # layout = nx.spring_layout(model.graph)
# # labels = nx.get_node_attributes(model.graph, "m")
# # #(model.graph, pos=layout, labels = labels) 
# # nx.draw(model.graph, pos=layout, labels = labels, arrows=True)
# # nx.is_directed(model.graph)
# # plt.show()
# #model.plot_network(model.graph)










