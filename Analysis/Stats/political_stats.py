import pandas as pd
import numpy as np
import os
from datetime import date

# Mapping for friendly names
FRIENDLY_NAMES = {
    'none_none': 'static',
    'random_none': 'random',
    'biased_same': 'local (similar)',
    'biased_diff': 'local (opposite)',
    'bridge_same': 'bridge (similar)',
    'bridge_diff': 'bridge (opposite)',
    'wtf_none': 'wtf',
    'node2vec_none': 'node2vec'
}

def get_friendly_name(mode, rewiring):
    """Get user-friendly algorithm name."""
    if mode is None: return "Unknown"
    
    if rewiring is None or pd.isna(rewiring) or rewiring == 'None':
        rewiring = "none"
    
    mode, rewiring = str(mode).lower(), str(rewiring).lower()
    
    key = f"{mode}_{rewiring}"
    if mode in ["none", "random", "wtf", "node2vec"]:
        key = f"{mode}_none"
    
    return FRIENDLY_NAMES.get(key, f"{mode} ({rewiring})")

def analyze_cooperativity_landscape(df, param_name='political_climate'):
    """Analyze cooperativity landscape across rewiring strategies and topologies."""
    # Add friendly name column
    df['friendly_name'] = df.apply(lambda r: get_friendly_name(r['mode'], r['rewiring']), axis=1)
    
    # Count positive vs total outcomes
    strategy_metrics = []
    for name, group in df.groupby('friendly_name'):
        total_count = len(group)
        positive_count = (group['state'] > 0).sum()
        positive_ratio = positive_count / total_count if total_count > 0 else 0
        
        # Calculate mean positive state (when positive)
        positive_df = group[group['state'] > 0]
        mean_positive = positive_df['state'].mean() if not positive_df.empty else np.nan
        
        strategy_metrics.append({
            'strategy': name,
            'total_count': total_count,
            'positive_count': positive_count,
            'positive_ratio': positive_ratio,
            'mean_positive_state': mean_positive,
            'is_opposite': '(opposite)' in name.lower()
        })
    
    # Analyze by topology
    topology_metrics = []
    for topology, group in df.groupby('topology'):
        total_count = len(group)
        positive_count = (group['state'] > 0).sum()
        positive_ratio = positive_count / total_count if total_count > 0 else 0
        
        # Calculate mean positive state (when positive)
        positive_df = group[group['state'] > 0]
        mean_positive = positive_df['state'].mean() if not positive_df.empty else np.nan
        
        # Check if this is a directed topology
        is_directed = topology in ['DPAH', 'Twitter']
        
        topology_metrics.append({
            'topology': topology,
            'total_count': total_count,
            'positive_count': positive_count,
            'positive_ratio': positive_ratio,
            'mean_positive_state': mean_positive,
            'is_directed': is_directed
        })
    
    # Analyze by topology + strategy combinations
    combo_metrics = []
    for (strategy, topology), group in df.groupby(['friendly_name', 'topology']):
        total_count = len(group)
        positive_count = (group['state'] > 0).sum()
        positive_ratio = positive_count / total_count if total_count > 0 else 0
        
        # Calculate mean positive state (when positive)
        positive_df = group[group['state'] > 0]
        mean_positive = positive_df['state'].mean() if not positive_df.empty else np.nan
        
        combo_metrics.append({
            'strategy': strategy,
            'topology': topology,
            'total_count': total_count,
            'positive_count': positive_count,
            'positive_ratio': positive_ratio, 
            'mean_positive_state': mean_positive
        })
    
    # Convert to DataFrames and sort by positive ratio
    strategy_df = pd.DataFrame(strategy_metrics).sort_values('positive_ratio', ascending=False)
    topology_df = pd.DataFrame(topology_metrics).sort_values('positive_ratio', ascending=False)
    combo_df = pd.DataFrame(combo_metrics).sort_values('positive_ratio', ascending=False)
    
    return strategy_df, topology_df, combo_df

def find_data_file():
    """Find a heatmap sweep file to analyze."""
    file_list = [f for f in os.listdir("../../Output") if f.endswith(".csv") and "heatmap_sweep" in f]
    
    if not file_list:
        print("No heatmap sweep files found.")
        return None
    
    for i, file in enumerate(file_list):
        print(f"{i}: {file}")
    
    try:
        file_idx = int(input("Enter index of file to analyze: "))
        if 0 <= file_idx < len(file_list):
            return os.path.join("../../Output", file_list[file_idx])
        print("Invalid index. Using most recent file.")
    except ValueError:
        print("Invalid input. Using most recent file.")
    
    file_list.sort(key=lambda f: os.path.getmtime(os.path.join("../../Output", f)), reverse=True)
    return os.path.join("../../Output", file_list[0])

def main():
    """Main execution function."""
    data_path = find_data_file()
    if not data_path:
        print("No data file found. Please run parameter sweep first.")
        return
    
    print(f"Analyzing data from: {data_path}")
    
    # Load and prepare data
    df = pd.read_csv(data_path)
    df['rewiring'] = df['rewiring'].fillna('none')
    df['mode'] = df['mode'].fillna('none')
    
    # Run analysis
    strategy_df, topology_df, combo_df = analyze_cooperativity_landscape(df)
    
    # Output key results
    pd.set_option('display.float_format', '{:.3f}'.format)
    
    print("\n=== Cooperativity Landscape by Strategy ===")
    print(strategy_df[['strategy', 'positive_ratio', 'positive_count', 'total_count', 'mean_positive_state']])
    
    print("\n=== Cooperativity Landscape by Topology ===")
    print(topology_df[['topology', 'positive_ratio', 'positive_count', 'total_count', 'mean_positive_state']])
    
    # Focus on opposite variants
    opposite_variants = strategy_df[strategy_df['is_opposite']]
    print("\n=== 'Opposite' Variant Performance ===")
    if not opposite_variants.empty:
        print(opposite_variants[['strategy', 'positive_ratio', 'positive_count', 'total_count', 'mean_positive_state']])
        
        # Extract values for placeholders
        local_opposite = opposite_variants[opposite_variants['strategy'] == 'local (opposite)']
        bridge_opposite = opposite_variants[opposite_variants['strategy'] == 'bridge (opposite)']
        
        if not local_opposite.empty and not bridge_opposite.empty:
            local_ratio = local_opposite.iloc[0]['positive_ratio']
            bridge_ratio = bridge_opposite.iloc[0]['positive_ratio']
            
            print(f"\nValues for manuscript placeholders:")
            print(f"Local (opposite) positive ratio: {local_ratio:.3f}")
            print(f"Bridge (opposite) positive ratio: {bridge_ratio:.3f}")
    else:
        print("No 'opposite' variants found in data.")
    
    # Compare directed vs undirected topologies
    directed = topology_df[topology_df['is_directed']]
    undirected = topology_df[~topology_df['is_directed']]
    
    if not directed.empty and not undirected.empty:
        print("\n=== Directed vs Undirected Topologies ===")
        print(f"Directed avg positive ratio: {directed['positive_ratio'].mean():.3f}")
        print(f"Undirected avg positive ratio: {undirected['positive_ratio'].mean():.3f}")
    
    # Save results
    output_dir = "../../Output"
    os.makedirs(output_dir, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    output_path = os.path.join(output_dir, f'cooperativity_landscape_analysis_{today}.csv')
    
    with open(output_path, 'w') as f:
        f.write("# Cooperativity Landscape Analysis\n\n")
        
        f.write("## Cooperativity by Strategy\n")
        strategy_df.to_csv(f, index=False)
        f.write("\n\n")
        
        f.write("## Cooperativity by Topology\n")
        topology_df.to_csv(f, index=False)
        f.write("\n\n")
        
        f.write("## Combined Results (Strategy x Topology)\n")
        combo_df.to_csv(f, index=False)
    
    print(f"\nAnalysis saved to: {output_path}")
    
    # Return values for placeholders
    if not opposite_variants.empty:
        local_opposite = opposite_variants[opposite_variants['strategy'] == 'local (opposite)']
        bridge_opposite = opposite_variants[opposite_variants['strategy'] == 'bridge (opposite)']
        
        value1 = f"{local_opposite.iloc[0]['positive_ratio']:.3f}" if not local_opposite.empty else "N/A"
        value2 = f"{bridge_opposite.iloc[0]['positive_ratio']:.3f}" if not bridge_opposite.empty else "N/A"
        
        return value1, value2
    
    return "N/A", "N/A"

if __name__ == "__main__":
    main()