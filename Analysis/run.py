
#
# Some notes on implementation for better or worse
# 
# Original program written by Sigrid Bratsberg, credit where credit is due! 
# 
# Switching from printing to a single file to making multiple out outs for a given
# set of input values is enabled by decommenting the newvar lines
# 
# TODO
# Someone should move all the parameters to the main program
#
#paramater anaylsis for rewiring values
#import pandas
import os
import pandas as pd 
from itertools import repeat
import time
import multiprocessing
import models_checks
import numpy as np 
from datetime import date
import glob
from itertools import product 

#assert 1 == 0


#random.seed(1574705741)    ## if you need to set a specific random seed put it here
#np.random.seed(1574705741)

date = date.today()
lock = None

#This is magic to present data racing I don't know why it works
def init(lock_):
    models_checks.init_lock(lock_)

if  __name__ ==  '__main__': 
    
    
    
    #Constants and Variables
    numberOfSimulations = 50
    numberOfProcessors =  int(0.8*multiprocessing.cpu_count()) # CPUs to use for parallelization

    start = time.time()
    lock = multiprocessing.Lock()
    
    pool = multiprocessing.Pool(processes=numberOfProcessors, initializer=init, initargs=(lock,))
    # ----------PATH TO SAVE FIGURES AND DATA-----------

    pathFig = '/Figs'
    pathData = '/Output'
    
    modelargs= models_checks.getargs()  # requires models.py to be imported
    #runs = 4   ## has to be even for multiple runs also n is actually n-1 because I'm lazy


    
    rewiring_list_h = ["diff", "same"]
    directed_topology_list = ["DPAH", "Twitter"]  
    undirected_topology_list = ["cl", "FB"]  
    
    # Create combined list for scenarios "biased" and "bridge" with "diff" and "same"
    # These can be on both directed and undirected networks
    combined_list1 = [(scenario, rewiring, topology)
                      for scenario in ["biased", "bridge"]
                      for rewiring in rewiring_list_h
                      for topology in directed_topology_list + undirected_topology_list]
    
    # Add combinations for "None" scenario with "node2vec" on all topologies
    # "node2vec" works on both directed and undirected
    combined_list2 = [("node2vec", "None", topology) for topology in directed_topology_list + undirected_topology_list]
    
    combined_list3 = [("None", "None", topology) for topology in directed_topology_list + undirected_topology_list]
    
    # Add combinations for "None" scenario with "wtf" only on directed topologies
    combined_list4 = [("wtf","None", topology) for topology in directed_topology_list]
    
    # Add combinations for "rand" scenario on all topologies
    combined_list_rand = [("random", "None", topology) for topology in directed_topology_list + undirected_topology_list]
    
    # Combine all lists
    combined_list = combined_list1 + combined_list_rand + combined_list2 + combined_list3 + combined_list4
    
    #combined_list = [("node2vec", "None", "DPAH")]
    

    out_list = []
    for i, v, k in combined_list:
      
     
       
        print("Started iteration: ", f"{i}_{v}_{k}")

        argList = []
        if k in "Twitter":
            top_file = "twitter_graph_N_789.gpickle"
            nwsize = 789
            
        elif k in "FB":
            top_file = "FB_graph_N_786.gpickle"
            nwsize = 786
        
        else:
            top_file = None
            nwsize = 800
        
        ## You can specify simulation parameters here. If they are not set here, they will default to some values set in models.py
        argList.append({"rewiringAlgorithm": i, "nwsize": nwsize, "rewiringMode": v, "type": k,
                        "top_file": top_file, "polarisingNode_f": 0.10, "timesteps": 60000 , "plot": False})
       
        
        #print (argList)
        
        for j in range(len(argList)):
            
            start_1  = time.time()
            sim = pool.starmap(models_checks.simulate, zip(range(numberOfSimulations), repeat(argList[j])))#, repeat(lock)))
        
            assert argList[0]["rewiringAlgorithm"] == str(sim[0].algo), "Inconsistent values"
            # #print(sim[0]. __class__. __name__)
            #print(sim[0].test) #sim[0].steps)
            
            fname = f'../Output/{i}_linkif_{v}_top_{j}.csv'
            print("starting save")
            out_list.append(models_checks.saveavgdata(sim, fname, args = argList[0]))
            end_1 = time.time()
            mins = (end_1 - start_1) / 60
            sec = (end_1 - start_1) % 60
            print(f'algorithim run is complete: {mins:5.0f} mins {round(sec)}s\n')
            
    pool.close()
    pool.join()
    
    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Runtime is complete: {round(mins/60)}) hours, {mins:5.0f} mins {round(sec)}s\n')
    
#assert 1 == 0
#%% post processing



    def process_outputs(out_list, nwsize):
        # Define the columns we want in our final output
        columns = ['t', 'avg_state', 'std_states', 'avgdegree', 'degreeSD', 'mindegree', 'maxdegree', 'scenario', 'rewiring', 'type']
    
        # Initialize a list to store processed DataFrames
        processed_dfs = []
    
        # Process each DataFrame in out_list
        for df in out_list:
            # Ensure all required columns are present
            for col in columns:
                if col not in df.columns:
                    df[col] = np.nan
            
            # Select and reorder columns
            df = df[columns]
            
            # Append to the list of processed DataFrames
            processed_dfs.append(df)
    
        # Concatenate all processed DataFrames at once
        combined_df = pd.concat(processed_dfs, ignore_index=True)
    
        # Optimize memory usage
        combined_df = combined_df.astype({
            't': 'int32',
            'avg_state': 'float32',
            'std_states': 'float32',
            'avgdegree': 'float32',
            'degreeSD': 'float32',
            'mindegree': 'float32',
            'maxdegree': 'float32',
            'scenario': 'category',
            'rewiring': 'category',
            'type': 'category'
        })
    
        # Save the combined DataFrame
        output_file = f'../Output/default_run_all_N_{nwsize}_n_{numberOfSimulations}_{date}.csv'
        combined_df.to_csv(output_file, index=False)
    
        print(f"Output saved to {output_file}")
    
        return combined_df



    combined_df = process_outputs(out_list, nwsize)
    
    for file in glob.glob("*embeddings*"):
        os.remove(file)

   