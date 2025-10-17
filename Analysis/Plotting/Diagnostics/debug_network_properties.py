"""
Debug script to analyze snapshot data and understand why network properties plots are empty.
Memory-safe version with streaming and monitoring for 16GB systems.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import gzip
import glob
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community as nx_community
import random
import psutil
import gc
import warnings
warnings.filterwarnings('ignore')

# Ultra-conservative memory management constants for 16GB system after crash
MAX_MEMORY_PERCENT = 60  # Don't exceed 60% of available memory (~9.6GB)
MEMORY_CHECK_INTERVAL = 5  # Check memory every 5 operations (more frequent)
SAFE_MEMORY_THRESHOLD = 75  # Stop processing if memory exceeds 75% (much more conservative)

def get_memory_usage():
    """Get current memory usage percentage"""
    return psutil.virtual_memory().percent

def check_memory_safety():
    """Check if memory usage is safe to continue"""
    memory_percent = get_memory_usage()
    if memory_percent > SAFE_MEMORY_THRESHOLD:
        print(f"⚠️ Memory usage at {memory_percent:.1f}% - stopping for safety")
        return False
    return True

def force_garbage_collection():
    """Force garbage collection and return memory freed"""
    before = psutil.virtual_memory().used
    gc.collect()
    after = psutil.virtual_memory().used
    freed_mb = (before - after) / 1024 / 1024
    return freed_mb

def sample_data(data, sample_fraction=0.1):
    """Sample a fraction of the data to reduce memory usage"""
    if isinstance(data, dict):
        # Sample scenarios
        scenarios = list(data.keys())
        n_sample = max(1, int(len(scenarios) * sample_fraction))
        sampled_scenarios = random.sample(scenarios, n_sample)
        
        sampled_data = {}
        for scenario in sampled_scenarios:
            scenario_data = data[scenario]
            if isinstance(scenario_data, list):
                # Sample runs within scenario
                n_runs_sample = max(1, int(len(scenario_data) * sample_fraction))
                sampled_data[scenario] = random.sample(scenario_data, n_runs_sample)
            else:
                sampled_data[scenario] = scenario_data
        
        return sampled_data
    return data

def load_and_sample_snapshot_data(data_dir="../../Output", sample_fraction=0.01, max_scenarios=1):
    """Ultra memory-safe streaming load and sample of snapshot data"""
    initial_memory = get_memory_usage()
    print(f"🔍 Looking for snapshot files... (Memory: {initial_memory:.1f}%)")
    
    snapshot_files = glob.glob(os.path.join(data_dir, "*_snapshots.pkl.gz"))
    snapshot_files.extend(glob.glob(os.path.join(data_dir, "all_snapshots*.pkl.gz")))
    
    if not snapshot_files:
        snapshot_files = glob.glob(os.path.join(data_dir, "*_snapshots.pkl"))
        snapshot_files.extend(glob.glob(os.path.join(data_dir, "all_snapshots*.pkl")))
    
    if not snapshot_files:
        print(f"❌ No snapshot files found in {data_dir}")
        return {}
    
    print(f"📁 Found {len(snapshot_files)} snapshot files")
    
    # Use smallest file first for safety
    file_sizes = [(f, os.path.getsize(f)) for f in snapshot_files]
    file_sizes.sort(key=lambda x: x[1])  # Sort by size
    
    file_path = file_sizes[0][0]  # Smallest file
    file_size_mb = file_sizes[0][1] / 1024 / 1024
    filename = os.path.basename(file_path)
    
    print(f"📂 Loading smallest file: {filename} ({file_size_mb:.1f} MB)")
    
    # Ultra conservative thresholds
    if file_size_mb > 500:  # > 500MB (reduced from 2GB)
        print(f"⚠️ File is {file_size_mb:.1f} MB - using ultra-minimal sample!")
        sample_fraction = 0.005  # 0.5%
        max_scenarios = 1
        print(f"   Reducing sample to {sample_fraction*100}% with max {max_scenarios} scenarios")
    
    try:
        print(f"💾 Memory before load: {get_memory_usage():.1f}%")
        
        # Load data
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        
        load_memory = get_memory_usage()
        print(f"📈 Memory after load: {load_memory:.1f}% (increase: {load_memory - initial_memory:.1f}%)")
        
        if load_memory > SAFE_MEMORY_THRESHOLD:
            print(f"⚠️ High memory usage after load - forcing garbage collection")
            freed_mb = force_garbage_collection()
            print(f"🗑️ Freed {freed_mb:.1f} MB")
        
        # Aggressively sample the data
        if isinstance(data, dict):
            print(f"📊 Original data has {len(data)} scenarios")
            
            # Limit to max_scenarios and sample within each
            scenarios = list(data.keys())[:max_scenarios]  # Take first N scenarios
            
            sampled_data = {}
            for scenario in scenarios:
                scenario_data = data[scenario]
                if isinstance(scenario_data, list):
                    # Take only first run for ultra-conservative memory usage
                    n_runs = min(1, len(scenario_data))  # Max 1 run only
                    selected_runs = scenario_data[:n_runs]
                    
                    # Sample timesteps within each run
                    processed_runs = []
                    for run in selected_runs:
                        if isinstance(run, dict):
                            if 'snapshots' in run:
                                snapshots = run['snapshots']
                            else:
                                snapshots = run
                            
                            # Keep only start and end timesteps for minimal memory usage
                            timesteps = list(snapshots.keys())
                            if len(timesteps) > 2:
                                # Keep only first and last timestep
                                keep_timesteps = [timesteps[0], timesteps[-1]]
                                filtered_snapshots = {t: snapshots[t] for t in keep_timesteps if t in snapshots}
                            else:
                                filtered_snapshots = snapshots
                            
                            if 'snapshots' in run:
                                processed_run = {'snapshots': filtered_snapshots}
                            else:
                                processed_run = filtered_snapshots
                            processed_runs.append(processed_run)
                    
                    sampled_data[scenario] = processed_runs
                else:
                    sampled_data[scenario] = scenario_data
            
            # Clear original data immediately
            del data
            force_garbage_collection()
            
            final_memory = get_memory_usage()
            print(f"✅ Sampled data has {len(sampled_data)} scenarios (Memory: {final_memory:.1f}%)")
            
            return sampled_data
        
        return data
    
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        force_garbage_collection()
        return {}

def debug_network_object(graph, label="graph"):
    """Debug a network object to understand its structure"""
    print(f"\n=== Debugging {label} ===")
    print(f"Type: {type(graph)}")
    print(f"Module: {graph.__class__.__module__ if hasattr(graph, '__class__') else 'N/A'}")
    
    # Check if it's a netin object
    if hasattr(graph, '__class__'):
        class_name = graph.__class__.__name__
        print(f"Class name: {class_name}")
        
        # Check for common netin generator methods
        if hasattr(graph, 'nodes'):
            try:
                nodes = list(graph.nodes())
                print(f"Number of nodes: {len(nodes)}")
                print(f"First few nodes: {nodes[:5]}")
            except Exception as e:
                print(f"Error accessing nodes: {e}")
        
        if hasattr(graph, 'edges'):
            try:
                edges = list(graph.edges())
                print(f"Number of edges: {len(edges)}")
                print(f"First few edges: {edges[:5]}")
            except Exception as e:
                print(f"Error accessing edges: {e}")
        
        # Test NetworkX functions directly
        print("\n--- Testing NetworkX functions ---")
        
        # Test clustering
        try:
            clustering = nx.average_clustering(graph)
            print(f"✓ Average clustering: {clustering}")
        except Exception as e:
            print(f"✗ Clustering error: {e}")
        
        # Test degree centrality
        try:
            degree_cent = nx.degree_centrality(graph)
            avg_degree_cent = np.mean(list(degree_cent.values()))
            print(f"✓ Average degree centrality: {avg_degree_cent}")
        except Exception as e:
            print(f"✗ Degree centrality error: {e}")
        
        # Test assortativity
        try:
            assortativity = nx.degree_assortativity_coefficient(graph)
            print(f"✓ Degree assortativity: {assortativity}")
        except Exception as e:
            print(f"✗ Assortativity error: {e}")
        
        # Test modularity
        try:
            communities = nx_community.louvain_communities(graph)
            modularity = nx_community.modularity(graph, communities)
            print(f"✓ Modularity: {modularity}")
        except Exception as e:
            print(f"✗ Modularity error: {e}")

def debug_data_structure(data):
    """Debug the overall data structure"""
    print("\n=== DATA STRUCTURE ANALYSIS ===")
    print(f"Top level type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Number of scenarios: {len(data)}")
        
        for i, (scenario_key, scenario_data) in enumerate(data.items()):
            if i >= 3:  # Only analyze first 3 scenarios
                break
                
            print(f"\n--- Scenario {i+1}: {scenario_key} ---")
            print(f"Scenario data type: {type(scenario_data)}")
            
            # Parse scenario information
            parts = scenario_key.split('_')
            network_type = parts[0] if parts else 'unknown'
            algorithm = parts[1] if len(parts) > 1 else 'none'
            rewiring_mode = parts[2] if len(parts) > 2 else 'none'
            
            print(f"Parsed - Network: {network_type}, Algorithm: {algorithm}, Mode: {rewiring_mode}")
            
            # Handle different data structures
            if isinstance(scenario_data, list):
                print(f"Number of runs: {len(scenario_data)}")
                
                # Analyze first run
                if scenario_data:
                    first_run = scenario_data[0]
                    print(f"First run type: {type(first_run)}")
                    
                    if isinstance(first_run, dict):
                        print(f"First run keys: {list(first_run.keys())}")
                        
                        # Check for snapshots
                        if 'snapshots' in first_run:
                            snapshots = first_run['snapshots']
                            print(f"Snapshots type: {type(snapshots)}")
                            print(f"Number of timesteps: {len(snapshots) if hasattr(snapshots, '__len__') else 'N/A'}")
                            
                            if hasattr(snapshots, 'keys'):
                                timesteps = list(snapshots.keys())
                                print(f"Timesteps: {timesteps}")
                                
                                # Debug first timestep
                                if timesteps:
                                    first_timestep = timesteps[0]
                                    first_graph = snapshots[first_timestep]
                                    debug_network_object(first_graph, f"timestep_{first_timestep}")
                        else:
                            # Direct snapshots structure
                            print(f"Direct snapshots with keys: {list(first_run.keys())[:10]}")
                            # Try to find numeric keys (timesteps)
                            numeric_keys = [k for k in first_run.keys() if isinstance(k, (int, float))]
                            if numeric_keys:
                                first_timestep = numeric_keys[0]
                                first_graph = first_run[first_timestep]
                                debug_network_object(first_graph, f"direct_timestep_{first_timestep}")
            else:
                # Single run scenario
                print("Single run scenario")
                if isinstance(scenario_data, dict):
                    print(f"Keys: {list(scenario_data.keys())[:10]}")

def test_property_calculation(data):
    """Memory-safe property calculation pipeline"""
    print(f"\n🧮 TESTING PROPERTY CALCULATION (Memory: {get_memory_usage():.1f}%)")
    
    all_data = []
    operation_count = 0
    
    for scenario_idx, (scenario, runs) in enumerate(data.items()):
        if not check_memory_safety():
            print("⚠️ Memory safety limit reached - stopping processing")
            break
            
        print(f"\n📊 Processing scenario {scenario_idx+1}/{len(data)}: {scenario}")
        
        # Parse scenario information
        parts = scenario.split('_')
        network_type = parts[0] if parts else 'unknown'
        algorithm = parts[1] if len(parts) > 1 else 'none'
        rewiring_mode = parts[2] if len(parts) > 2 else 'none'
        
        # Handle both list of runs and single run data
        runs_to_process = runs if isinstance(runs, list) else [runs]
        
        for run_idx, run_data in enumerate(runs_to_process):
            if not check_memory_safety():
                print("⚠️ Memory safety limit reached - stopping run processing")
                break
                
            print(f"   🔄 Processing run {run_idx+1}/{len(runs_to_process)}")
            
            if isinstance(run_data, dict):
                # Check if this is the metadata wrapper structure
                if 'snapshots' in run_data:
                    snapshots = run_data['snapshots']
                    print(f"      📦 Found metadata wrapper with {len(snapshots)} timesteps")
                else:
                    snapshots = run_data
                    print(f"      📦 Found direct snapshots dict with {len(snapshots)} timesteps")
                
                # Process timesteps with memory monitoring
                for timestep_idx, (timestep, graph) in enumerate(snapshots.items()):
                    operation_count += 1
                    
                    # Check memory every N operations
                    if operation_count % MEMORY_CHECK_INTERVAL == 0:
                        if not check_memory_safety():
                            print("⚠️ Memory safety limit reached - stopping timestep processing")
                            break
                        
                        # Force garbage collection periodically
                        if operation_count % (MEMORY_CHECK_INTERVAL * 2) == 0:
                            freed_mb = force_garbage_collection()
                            if freed_mb > 10:  # Only report if significant
                                print(f"      🗑️ Freed {freed_mb:.1f} MB (Memory: {get_memory_usage():.1f}%)")
                    
                    print(f"      ⏱️ Processing timestep {timestep} ({timestep_idx+1}/{len(snapshots)})")
                    
                    try:
                        # Quick node/edge count check first
                        if not hasattr(graph, 'nodes') or not hasattr(graph, 'edges'):
                            print(f"        ❌ Invalid graph object")
                            continue
                        
                        num_nodes = len(list(graph.nodes()))
                        num_edges = len(list(graph.edges()))
                        
                        if num_nodes == 0:
                            print(f"        ⚠️ Empty graph - skipping")
                            continue
                        
                        print(f"        📈 Graph: {num_nodes} nodes, {num_edges} edges")
                        
                        # Calculate properties with error handling
                        clustering = np.nan
                        assortativity = np.nan
                        
                        try:
                            clustering = nx.average_clustering(graph)
                        except Exception as e:
                            print(f"        ⚠️ Clustering calculation failed: {e}")
                        
                        try:
                            assortativity = nx.degree_assortativity_coefficient(graph)
                        except Exception as e:
                            print(f"        ⚠️ Assortativity calculation failed: {e}")
                        
                        properties = {
                            'timestep': timestep,
                            'run': run_idx,
                            'scenario': scenario,
                            'network_type': network_type,
                            'algorithm': algorithm,
                            'rewiring_mode': rewiring_mode,
                            'scenario_grouped': f"{algorithm}_{rewiring_mode}",
                            'clustering': clustering,
                            'assortativity': assortativity,
                            'modularity': np.nan,  # Skip modularity for memory safety
                            'gini_degree': np.nan,  # Skip gini for memory safety
                            'num_nodes': num_nodes,
                            'num_edges': num_edges
                        }
                        
                        all_data.append(properties)
                        print(f"        ✅ C: {clustering:.4f}, A: {assortativity:.4f}")
                        
                    except Exception as e:
                        print(f"        ❌ Error processing timestep {timestep}: {e}")
                        continue
                
                # Clear snapshots after processing to free memory
                if 'snapshots' in run_data:
                    run_data['snapshots'].clear()
                else:
                    run_data.clear()
    
    # Force final cleanup
    force_garbage_collection()
    final_memory = get_memory_usage()
    
    print(f"\n📊 Total data points processed: {len(all_data)} (Final memory: {final_memory:.1f}%)")
    
    if all_data:
        df = pd.DataFrame(all_data)
        print(f"\n✅ DataFrame shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Network types: {df['network_type'].unique()}")
        print(f"   Algorithms: {df['algorithm'].unique()}")
        print(f"   Scenarios: {df['scenario_grouped'].unique()}")
        
        # Check for NaN values
        print(f"\n🔍 Data quality:")
        nan_counts = df[['clustering', 'assortativity']].isna().sum()
        print(f"   NaN clustering: {nan_counts['clustering']}/{len(df)} ({100*nan_counts['clustering']/len(df):.1f}%)")
        print(f"   NaN assortativity: {nan_counts['assortativity']}/{len(df)} ({100*nan_counts['assortativity']/len(df):.1f}%)")
        
        # Show sample data
        print(f"\n📋 Sample data:")
        print(df[['scenario', 'timestep', 'clustering', 'assortativity', 'num_nodes', 'num_edges']].head())
        
        return df
    else:
        print("❌ No data processed successfully!")
        return pd.DataFrame()

def main():
    """Memory-safe main debugging function for 16GB systems"""
    print("🚀 MEMORY-SAFE NETWORK PROPERTIES DEBUG SCRIPT")
    print(f"💾 System: Ubuntu 25.04, 16GB RAM, 8 cores")
    print(f"📊 Initial memory usage: {get_memory_usage():.1f}%")
    print("="*60)
    
    # Set random seed for reproducible sampling
    random.seed(42)
    
    # Use ultra-conservative settings for 16GB system after crash
    print("🔧 Loading ultra-minimal sample of snapshot data for safety...")
    data = load_and_sample_snapshot_data(
        sample_fraction=0.01,  # Only 1% to prevent crashes
        max_scenarios=1        # Only 1 scenario
    )
    
    if not data:
        print("❌ No data loaded. Exiting.")
        return
    
    print(f"💾 Memory after data load: {get_memory_usage():.1f}%")
    
    # Debug data structure (but limit output)
    print("\n" + "="*60)
    debug_data_structure(data)
    
    # Test property calculation with memory monitoring
    print("\n" + "="*60)
    df = test_property_calculation(data)
    
    # Save sample results
    if not df.empty:
        output_file = "debug_network_properties_sample.csv"
        print(f"\n💾 Saving results to {output_file}...")
        df.to_csv(output_file, index=False)
        
        file_size_kb = os.path.getsize(output_file) / 1024
        print(f"✅ Sample data saved: {output_file} ({file_size_kb:.1f} KB)")
        print(f"   Data points: {len(df)}")
        print(f"   Scenarios: {df['scenario'].nunique()}")
        print(f"   Timesteps: {df['timestep'].nunique()}")
    else:
        print("❌ No data to save")
    
    # Final cleanup and memory report
    del data
    if 'df' in locals():
        del df
    final_freed = force_garbage_collection()
    final_memory = get_memory_usage()
    
    print(f"\n🏁 FINAL REPORT:")
    print(f"   Final memory usage: {final_memory:.1f}%")
    print(f"   Memory freed in cleanup: {final_freed:.1f} MB")
    print(f"   Status: {'✅ SUCCESS' if final_memory < 80 else '⚠️ HIGH MEMORY'}")

if __name__ == "__main__":
    main()