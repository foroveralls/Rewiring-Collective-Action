#!/usr/bin/env python3
"""
Three-panel visualization showing algorithm performance across stubbornness regimes
Matches manuscript style for PNAS/Nature Communications submission
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

cm = 1/2.54
FONT_SIZE = 6

# Color scheme matching existing plots
FRIENDLY_COLORS = {
    'static': '#EE7733', 'random': '#0077BB', 'local (similar)': '#AA4499', 
    'local (opposite)': '#117733', 'bridge (similar)': '#CC3311', 'bridge (opposite)': '#EE3377',
    'empirical wtf': '#BBBBBB', 'empirical node2vec': '#44BB99'
}

# Marker styles for topologies (matching existing style)
TOPOLOGY_MARKERS = {'DPAH': 'x', 'cl': '+', 'Twitter': '^', 'FB': 'o'}

def setup_style():
    """Set up matplotlib style to match manuscript standards"""
    plt.rcParams.update({
        'font.size': FONT_SIZE, 
        'pdf.fonttype': 42, 
        'ps.fonttype': 42,
        'axes.linewidth': 0.5, 
        'xtick.major.width': 0.5, 
        'ytick.major.width': 0.5,
        'xtick.labelsize': FONT_SIZE-1, 
        'ytick.labelsize': FONT_SIZE-1,
        'axes.labelsize': FONT_SIZE-1, 
        'axes.titlesize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE-2
    })

def load_regime_data():
    """Load and prepare regime-based data from CSV files - DIRECTED NETWORKS ONLY"""
    # Load topology summaries for each regime
    base_dir = "../../Output/Stats/stubborness_backfirer"
    
    # Define directed network topologies for fair comparison
    DIRECTED_TOPOLOGIES = ['FB', 'Twitter']
    
    regime_data = {}
    for regime in ['low', 'medium', 'high']:
        file_path = f"{base_dir}/topology_summary_{regime}_regime_20250925.csv"
        if not os.path.exists(file_path):
            file_path = f"{base_dir}/topology_summary_{regime}_regime_20250924.csv"
            
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Filter for directed networks only
            df_filtered = df[df['topology'].isin(DIRECTED_TOPOLOGIES)]
            regime_data[regime] = df_filtered
            print(f"Loaded {len(df_filtered)} directed topologies for {regime} regime: {list(df_filtered['topology'])}")
        else:
            print(f"Warning: {file_path} not found")
    
    return regime_data

def prepare_algorithm_data():
    """Prepare algorithm-level data from comprehensive comparison files (ALL NETWORKS)"""
    base_dir = "../../Output/Stats/stubborness_backfirer"
    
    algorithm_data = {}
    regime_mapping = {'low': 'low', 'medium': 'medium', 'high': 'high'}
    
    for regime_key, regime_name in regime_mapping.items():
        # Use non-FIXED files to get complete data including WTF and Node2Vec
        file_path = f"{base_dir}/comprehensive_algorithm_comparison_{regime_key}_20250925.csv"
        if not os.path.exists(file_path):
            file_path = f"{base_dir}/comprehensive_algorithm_comparison_{regime_key}_20250924.csv"
            
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Restructure data for plotting
            algorithms = []
            cooperation = []
            polarization = []
            coop_volume = []
            backfirer_fraction = []
            
            for col in df.columns[1:]:  # Skip 'Metric' column
                # Get values for each algorithm, handling NaN values
                try:
                    coop_val = df[df['Metric'] == 'Mean Cooperation'][col].values[0]
                    polar_val = df[df['Metric'] == 'Mean Polarization'][col].values[0]
                    volume_val = df[df['Metric'] == 'Cooperative Volume %'][col].values[0]
                    backfirer_val = df[df['Metric'] == 'Mean Backfirer Fraction'][col].values[0]
                    
                    # Only include if cooperation value is not NaN/empty
                    if pd.notna(coop_val) and coop_val != '':
                        algorithms.append(col)
                        cooperation.append(coop_val)
                        polarization.append(polar_val)
                        coop_volume.append(volume_val)
                        backfirer_fraction.append(backfirer_val)
                except (IndexError, ValueError):
                    # Skip if any values are missing/invalid
                    continue
            
            algorithm_data[regime_name] = {
                'algorithms': algorithms,
                'cooperation': cooperation,
                'polarization': polarization,
                'cooperative_volume': coop_volume,
                'backfirer_fraction': backfirer_fraction
            }
            
            print(f"Loaded {len(algorithms)} algorithms for {regime_name} regime: {algorithms}")
    
    return algorithm_data

def create_three_panel_plot(algorithm_data):
    """Create three-panel plot showing regime performance"""
    setup_style()
    
    # Create figure with two panels (combining cooperation and polarization)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12*cm, 6*cm))
    
    regimes = ['low', 'medium', 'high']
    regime_labels = ['Low\n(< 0.4)', 'Medium\n(0.4-0.7)', 'High\n(> 0.7)']
    x_pos = np.arange(len(regimes))
    
    # Get all algorithms (use medium regime as reference)
    algorithms = algorithm_data['medium']['algorithms']
    
    # Create color mapping for algorithms - group by bias type as shown in user's image
    algorithm_colors = {}
    for alg in algorithms:
        alg_lower = alg.lower()
        if alg_lower == 'opposite':
            algorithm_colors[alg] = FRIENDLY_COLORS['local (opposite)']  # Green
        elif alg_lower == 'similar':
            algorithm_colors[alg] = FRIENDLY_COLORS['local (similar)']   # Purple  
        elif alg_lower == 'wtf':
            algorithm_colors[alg] = FRIENDLY_COLORS['empirical wtf']     # Gray
        elif alg_lower == 'node2vec':
            algorithm_colors[alg] = FRIENDLY_COLORS['empirical node2vec'] # Teal
        elif alg_lower == 'static':
            algorithm_colors[alg] = FRIENDLY_COLORS['static']            # Orange
        elif alg_lower == 'random':
            algorithm_colors[alg] = FRIENDLY_COLORS['random']            # Blue
        else:
            algorithm_colors[alg] = '#666666'  # Default gray
    
    # Create twin axis for Panel A (polarization)
    ax1_twin = ax1.twinx()
    
    # Plot each algorithm across regimes
    for i, alg in enumerate(algorithms):
        # Extract values for this algorithm across regimes
        coop_vals = [algorithm_data[regime]['cooperation'][i] for regime in regimes]
        polar_vals = [algorithm_data[regime]['polarization'][i] for regime in regimes]
        volume_vals = [algorithm_data[regime]['cooperative_volume'][i] for regime in regimes]
        
        color = algorithm_colors.get(alg, '#666666')
        
        # Panel A: Mean Cooperation (solid lines) and Polarization (dotted lines)
        ax1.plot(x_pos, coop_vals, 'o-', color=color, linewidth=1, 
                markersize=4, alpha=0.8, label=alg)
        ax1_twin.plot(x_pos, polar_vals, 'o:', color=color, linewidth=1, 
                markersize=4, alpha=0.6)
        
        # Panel B: Cooperative Volume
        ax2.plot(x_pos, volume_vals, 'o-', color=color, linewidth=1, 
                markersize=4, alpha=0.8)
    
    # Customize Panel A - Cooperation & Polarization Combined with dual y-axes
    ax1.set_xlabel('Stubbornness Regime', fontsize=FONT_SIZE-1, labelpad=2)
    ax1.set_ylabel('$a$', fontsize=FONT_SIZE-1, labelpad=2, color='black')
    ax1_twin.set_ylabel('$\sigma(a)$', fontsize=FONT_SIZE-1, labelpad=2, color='black')
    ax1.set_title('A', fontsize=FONT_SIZE, pad=5, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(regime_labels, fontsize=FONT_SIZE-2)
    ax1.grid(True, alpha=0.3, linewidth=0.3)
    ax1.tick_params(labelsize=FONT_SIZE-2)
    ax1_twin.tick_params(labelsize=FONT_SIZE-2)
    
    # Customize Panel B - Cooperative Volume
    ax2.set_xlabel('Stubbornness Regime', fontsize=FONT_SIZE-1, labelpad=2)
    ax2.set_ylabel('Cooperative Volume (%)', fontsize=FONT_SIZE-1, labelpad=2)
    ax2.set_title('B', fontsize=FONT_SIZE, pad=5, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(regime_labels, fontsize=FONT_SIZE-2)
    ax2.grid(True, alpha=0.3, linewidth=0.3)
    ax2.tick_params(labelsize=FONT_SIZE-2)
    
    # Add legend at the bottom
    fig.legend(bbox_to_anchor=(0.5, 0.02), loc='upper center', ncol=4,
              fontsize=FONT_SIZE-2, frameon=True, fancybox=False, 
              edgecolor='black', facecolor='white')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)  # Make room for legend
    
    return fig

def create_algorithm_focus_plot(algorithm_data):
    """Create plot highlighting specific algorithms of interest (WTF vs best performers)"""
    setup_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12*cm, 6*cm))
    
    regimes = ['low', 'medium', 'high']
    regime_labels = ['Low', 'Medium', 'High']
    x_pos = np.arange(len(regimes))
    
    # Focus on key algorithms for the narrative
    focus_algorithms = ['Opposite', 'WTF', 'Similar', 'Static']
    
    algorithms = algorithm_data['medium']['algorithms']
    
    # Create twin axis for Panel A (polarization)
    ax1_twin = ax1.twinx()
    
    for alg in focus_algorithms:
        if alg in algorithms:
            alg_idx = algorithms.index(alg)
            
            # Extract values across regimes
            coop_vals = [algorithm_data[regime]['cooperation'][alg_idx] for regime in regimes]
            polar_vals = [algorithm_data[regime]['polarization'][alg_idx] for regime in regimes]  
            volume_vals = [algorithm_data[regime]['cooperative_volume'][alg_idx] for regime in regimes]
            
            # Set color and style based on algorithm type
            if alg == 'WTF':
                color = FRIENDLY_COLORS['empirical wtf']
                linestyle = '--'
                linewidth = 2
                markersize = 6
            elif alg == 'Opposite':
                color = FRIENDLY_COLORS['local (opposite)']
                linestyle = '-'
                linewidth = 2
                markersize = 5
            elif alg == 'Similar':
                color = FRIENDLY_COLORS['local (similar)']
                linestyle = '-'
                linewidth = 1.5
                markersize = 4
            elif alg == 'Static':
                color = FRIENDLY_COLORS['static']
                linestyle = '-'
                linewidth = 1.5
                markersize = 4
            else:
                color = '#666666'
                linestyle = '-'
                linewidth = 1.5
                markersize = 4
            
            # Plot on panels
            line_fmt = 'o--' if linestyle == '--' else 'o-'
            ax1.plot(x_pos, coop_vals, line_fmt, color=color, linewidth=linewidth,
                    markersize=markersize, alpha=0.9, label=alg)
            ax1_twin.plot(x_pos, polar_vals, 'o:', color=color, linewidth=linewidth,
                    markersize=markersize, alpha=0.6)
            
            ax2.plot(x_pos, volume_vals, line_fmt, color=color, linewidth=linewidth,
                    markersize=markersize, alpha=0.9)
    
    # Customize panels
    for ax, ylabel, title in zip([ax1, ax2], 
                                ['$a$', 'Cooperative Volume (%)'],
                                ['A', 'B']):
        ax.set_xlabel('Stubbornness Regime', fontsize=FONT_SIZE-1)
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE-1)
        ax.set_title(title, fontsize=FONT_SIZE, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(regime_labels, fontsize=FONT_SIZE-1)
        ax.grid(True, alpha=0.3, linewidth=0.3)
        ax.tick_params(labelsize=FONT_SIZE-2)
    
    # Add polarization label to twin axis
    ax1_twin.set_ylabel('$\sigma(a)$', fontsize=FONT_SIZE-1, color='black')
    ax1_twin.tick_params(labelsize=FONT_SIZE-2)
    
    # Legend at the bottom
    fig.legend(bbox_to_anchor=(0.5, 0.02), loc='upper center', ncol=2,
              fontsize=FONT_SIZE-1, frameon=True, fancybox=False,
              edgecolor='black', facecolor='white')
    
    plt.subplots_adjust(bottom=0.18)  # Make room for legend
    
    plt.tight_layout()
    return fig

def prepare_backfirer_regime_data():
    """Prepare algorithm data organized by backfirer fraction levels using comprehensive comparison files"""
    base_dir = "../../Output/Stats/stubborness_backfirer"
    
    # Load all regime data first - use comprehensive files for consistency
    all_regime_data = {}
    for regime_key in ['low', 'medium', 'high']:
        file_path = f"{base_dir}/comprehensive_algorithm_comparison_{regime_key}_20250925.csv"
        if not os.path.exists(file_path):
            file_path = f"{base_dir}/comprehensive_algorithm_comparison_{regime_key}_20250924.csv"
            
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_regime_data[regime_key] = df
    
    # Get all algorithms and their backfirer fractions across regimes
    if 'medium' not in all_regime_data:
        return {}, 0, 0
    
    algorithms = [col for col in all_regime_data['medium'].columns[1:] if col in ['Opposite', 'Similar', 'WTF', 'Node2Vec', 'Static', 'Random']]
    all_backfirer_fractions = []
    
    # Collect all backfirer fraction values to determine thresholds
    for alg in algorithms:
        for regime in ['low', 'medium', 'high']:
            if regime in all_regime_data:
                df = all_regime_data[regime]
                try:
                    backfirer_val = df[df['Metric'] == 'Mean Backfirer Fraction'][alg].values[0]
                    if pd.notna(backfirer_val) and backfirer_val != '':
                        all_backfirer_fractions.append((backfirer_val, alg, regime))
                except (IndexError, KeyError):
                    continue
    
    # Sort by backfirer fraction
    all_backfirer_fractions.sort(key=lambda x: x[0])
    
    # With non-FIXED data, backfirer fractions are much smaller (cooperative-only range ~0.004-0.138)
    # Use data-driven thresholds based on actual values
    if all_backfirer_fractions:
        values = [x[0] for x in all_backfirer_fractions]
        min_val, max_val = min(values), max(values)
        
        # Use tertiles to split into three regimes
        low_threshold = min_val + (max_val - min_val) / 3
        high_threshold = min_val + 2 * (max_val - min_val) / 3
    else:
        # Fallback thresholds 
        low_threshold = 0.05
        high_threshold = 0.10
    
    print(f"Using updated backfirer thresholds: Low ≤ {low_threshold}, Medium {low_threshold}-{high_threshold}, High > {high_threshold}")
    
    # Organize data with same structure as stubbornness regimes
    backfirer_regime_data = {}
    
    for regime_name in ['low_backfirer', 'medium_backfirer', 'high_backfirer']:
        backfirer_regime_data[regime_name] = {
            'algorithms': algorithms.copy(),
            'cooperation': [],
            'polarization': [],
            'cooperative_volume': []
        }
    
    # For each algorithm, collect its metrics across the stubbornness regimes
    # and classify each data point into backfirer regimes
    for i, alg in enumerate(algorithms):
        regime_data = {'low_backfirer': [], 'medium_backfirer': [], 'high_backfirer': []}
        
        for regime in ['low', 'medium', 'high']:
            if regime in all_regime_data:
                df = all_regime_data[regime]
                try:
                    backfirer_val = df[df['Metric'] == 'Mean Backfirer Fraction'][alg].values[0]
                    coop_val = df[df['Metric'] == 'Mean Cooperation'][alg].values[0]
                    polar_val = df[df['Metric'] == 'Mean Polarization'][alg].values[0] 
                    volume_val = df[df['Metric'] == 'Cooperative Volume %'][alg].values[0]
                    
                    # Only include if values are valid
                    if pd.notna(coop_val) and coop_val != '' and pd.notna(backfirer_val) and backfirer_val != '':
                        # Classify this data point into backfirer regime
                        if backfirer_val <= low_threshold:
                            regime_key = 'low_backfirer'
                        elif backfirer_val <= high_threshold:
                            regime_key = 'medium_backfirer'
                        else:
                            regime_key = 'high_backfirer'
                        
                        regime_data[regime_key].append({
                            'cooperation': coop_val,
                            'polarization': polar_val,
                            'cooperative_volume': volume_val
                        })
                except (IndexError, KeyError):
                    continue
        
        # Average values for each backfirer regime for this algorithm
        for regime_key in ['low_backfirer', 'medium_backfirer', 'high_backfirer']:
            if regime_data[regime_key]:
                avg_coop = np.mean([d['cooperation'] for d in regime_data[regime_key]])
                avg_polar = np.mean([d['polarization'] for d in regime_data[regime_key]])
                avg_volume = np.mean([d['cooperative_volume'] for d in regime_data[regime_key]])
            else:
                # If no data points fall in this regime, use NaN
                avg_coop = np.nan
                avg_polar = np.nan
                avg_volume = np.nan
            
            backfirer_regime_data[regime_key]['cooperation'].append(avg_coop)
            backfirer_regime_data[regime_key]['polarization'].append(avg_polar)
            backfirer_regime_data[regime_key]['cooperative_volume'].append(avg_volume)
    
    return backfirer_regime_data, low_threshold, high_threshold

def create_backfirer_regime_plot(backfirer_data, low_threshold, high_threshold):
    """Create plot showing performance across backfirer regimes"""
    setup_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12*cm, 6*cm))
    
    regimes = ['low_backfirer', 'medium_backfirer', 'high_backfirer']
    regime_labels = [f'Low\n(≤ {low_threshold:.3f})', 
                     f'Medium\n({low_threshold:.3f}-{high_threshold:.3f})', 
                     f'High\n(> {high_threshold:.3f})']
    x_pos = np.arange(len(regimes))
    
    # Create twin axis for Panel A (polarization)
    ax1_twin = ax1.twinx()
    
    # Get all algorithms (use any regime as reference since they all have same algorithms)
    algorithms = backfirer_data['low_backfirer']['algorithms']
    
    # Create color mapping for algorithms (handling detailed scenario names)
    algorithm_colors = {}
    for alg in algorithms:
        alg_lower = alg.lower()
        if 'biased' in alg_lower or 'opposite' in alg_lower:
            algorithm_colors[alg] = FRIENDLY_COLORS['local (opposite)']
        elif 'similar' in alg_lower:
            algorithm_colors[alg] = FRIENDLY_COLORS['local (similar)']
        elif 'bridge' in alg_lower:
            algorithm_colors[alg] = FRIENDLY_COLORS['bridge (opposite)']
        elif 'wtf' in alg_lower:
            algorithm_colors[alg] = FRIENDLY_COLORS['empirical wtf']
        elif 'node2vec' in alg_lower:
            algorithm_colors[alg] = FRIENDLY_COLORS['empirical node2vec']
        elif 'static' in alg_lower:
            algorithm_colors[alg] = FRIENDLY_COLORS['static']
        elif 'random' in alg_lower:
            algorithm_colors[alg] = FRIENDLY_COLORS['random']
        else:
            algorithm_colors[alg] = '#666666'  # Default gray
    
    # Plot each algorithm across backfirer regimes (same structure as stubbornness regimes)
    for i, alg in enumerate(algorithms):
        # Extract values for this algorithm across backfirer regimes
        coop_vals = [backfirer_data[regime]['cooperation'][i] for regime in regimes]
        polar_vals = [backfirer_data[regime]['polarization'][i] for regime in regimes]
        volume_vals = [backfirer_data[regime]['cooperative_volume'][i] for regime in regimes]
        
        # Skip algorithms with all NaN values
        if all(np.isnan(val) for val in coop_vals):
            continue
            
        color = algorithm_colors.get(alg, '#666666')
        
        # Panel A: Mean Cooperation (solid lines) and Polarization (dotted lines)
        ax1.plot(x_pos, coop_vals, 'o-', color=color, linewidth=1, 
                markersize=4, alpha=0.8, label=alg)
        ax1_twin.plot(x_pos, polar_vals, 'o:', color=color, linewidth=1, 
                markersize=4, alpha=0.6)
        
        # Panel B: Cooperative Volume
        ax2.plot(x_pos, volume_vals, 'o-', color=color, linewidth=1, 
                markersize=4, alpha=0.8)
    
    # Customize Panel A - Cooperation & Polarization Combined with dual y-axes
    ax1.set_xlabel('Backfirer Regime', fontsize=FONT_SIZE-1, labelpad=2)
    ax1.set_ylabel('$a$', fontsize=FONT_SIZE-1, labelpad=2, color='black')
    ax1_twin.set_ylabel('$\sigma(a)$', fontsize=FONT_SIZE-1, labelpad=2, color='black')
    ax1.set_title('A', fontsize=FONT_SIZE, pad=5, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(regime_labels, fontsize=FONT_SIZE-2)
    ax1.grid(True, alpha=0.3, linewidth=0.3)
    ax1.tick_params(labelsize=FONT_SIZE-2)
    ax1_twin.tick_params(labelsize=FONT_SIZE-2)
    
    # Customize Panel B - Cooperative Volume
    ax2.set_xlabel('Backfirer Regime', fontsize=FONT_SIZE-1, labelpad=2)
    ax2.set_ylabel('Cooperative Volume (%)', fontsize=FONT_SIZE-1, labelpad=2)
    ax2.set_title('B', fontsize=FONT_SIZE, pad=5, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(regime_labels, fontsize=FONT_SIZE-2)
    ax2.grid(True, alpha=0.3, linewidth=0.3)
    ax2.tick_params(labelsize=FONT_SIZE-2)
    
    # Add legend at the bottom
    fig.legend(bbox_to_anchor=(0.5, 0.02), loc='upper center', ncol=4,
              fontsize=FONT_SIZE-2, frameon=True, fancybox=False, 
              edgecolor='black', facecolor='white')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)  # Make room for legend
    
    return fig

def main():
    """Generate regime performance visualization (ALL NETWORKS - main analysis)"""
    print("Creating regime performance visualization for ALL networks...")
    
    # Load algorithm data
    algorithm_data = prepare_algorithm_data()
    
    if not algorithm_data:
        print("No data found. Please check comprehensive algorithm comparison files.")
        return
    
    print(f"Loaded algorithm data for regimes: {list(algorithm_data.keys())}")
    
    # Create output directory
    output_dir = "../../Figs/Regime_Analysis"
    os.makedirs(output_dir, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    
    # Generate comprehensive three-panel plot
    print("Creating comprehensive three-panel plot...")
    fig1 = create_three_panel_plot(algorithm_data)
    output_path1 = f"{output_dir}/regime_performance_comprehensive_{today}.pdf"
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"Comprehensive plot saved: {output_path1}")
    plt.show()
    plt.close()
    
    # Generate focused plot for key algorithms
    print("Creating focused algorithm comparison...")
    fig2 = create_algorithm_focus_plot(algorithm_data)
    output_path2 = f"{output_dir}/regime_performance_focused_{today}.pdf"
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Focused plot saved: {output_path2}")
    plt.show()
    plt.close()
    
    # Generate backfirer regime plot
    print("Creating backfirer regime analysis...")
    backfirer_data, low_thresh, high_thresh = prepare_backfirer_regime_data()
    fig3 = create_backfirer_regime_plot(backfirer_data, low_thresh, high_thresh)
    output_path3 = f"{output_dir}/regime_backfirer_regimes_{today}.pdf"
    fig3.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"Backfirer regime plot saved: {output_path3}")
    plt.show()
    plt.close()
    
    print("Regime visualization complete!")
    
    # Print some key insights
    print("\n=== KEY INSIGHTS ===")
    for regime in ['low', 'medium', 'high']:
        if regime in algorithm_data:
            algorithms = algorithm_data[regime]['algorithms']
            cooperation = algorithm_data[regime]['cooperation']
            
            # Find best and worst performers
            best_idx = np.argmax(cooperation)
            worst_idx = np.argmin(cooperation)
            
            print(f"{regime.capitalize()} regime:")
            print(f"  Best: {algorithms[best_idx]} ({cooperation[best_idx]:.3f})")
            print(f"  Worst: {algorithms[worst_idx]} ({cooperation[worst_idx]:.3f})")
            
            # Find WTF performance
            if 'WTF' in algorithms:
                wtf_idx = algorithms.index('WTF')
                wtf_rank = sorted(range(len(cooperation)), key=lambda i: cooperation[i], reverse=True).index(wtf_idx) + 1
                print(f"  WTF: {algorithms[wtf_idx]} ({cooperation[wtf_idx]:.3f}) - Rank {wtf_rank}/{len(algorithms)}")

def prepare_directed_algorithm_data():
    """Prepare algorithm-level data from detailed metrics - DIRECTED NETWORKS ONLY (DPAH + Twitter)"""
    base_dir = "../../Output/Stats/stubborness_backfirer"
    
    # Define directed network topologies for supplementary analysis
    DIRECTED_TOPOLOGIES = ['DPAH', 'Twitter']
    
    # Load detailed metrics file
    file_path = f"{base_dir}/regime_based_metrics_20250925.csv"
    if not os.path.exists(file_path):
        print(f"Detailed metrics file not found: {file_path}")
        return {}
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} total records from detailed metrics")
    
    # Filter for directed networks only
    df_directed = df[df['topology'].isin(DIRECTED_TOPOLOGIES)]
    print(f"Filtered to {len(df_directed)} records for directed networks: {DIRECTED_TOPOLOGIES}")
    
    algorithm_data = {}
    
    for regime in ['low', 'medium', 'high']:
        # Filter data for this regime and directed topologies
        regime_data = df_directed[df_directed['stubbornness_regime'] == regime]
        
        if regime_data.empty:
            print(f"No data found for {regime} regime in directed networks")
            continue
        
        # Group by algorithm (scenario) and calculate means across directed topologies
        algorithm_stats = regime_data.groupby('scenario').agg({
            'mean_cooperation': 'mean',
            'mean_polarization': 'mean',
            'cooperative_volume_percent': 'mean',
            'mean_backfirer_fraction': 'mean'
        }).reset_index()
        
        algorithm_data[regime] = {
            'algorithms': list(algorithm_stats['scenario']),
            'cooperation': list(algorithm_stats['mean_cooperation']),
            'polarization': list(algorithm_stats['mean_polarization']),
            'cooperative_volume': list(algorithm_stats['cooperative_volume_percent']),
            'backfirer_fraction': list(algorithm_stats['mean_backfirer_fraction'])
        }
        
        print(f"Processed {len(algorithm_stats)} algorithms for {regime} regime: {list(algorithm_stats['scenario'])}")
    
    return algorithm_data

def main_directed_only():
    """Generate regime performance visualization - DIRECTED NETWORKS ONLY (supplementary analysis)"""
    print("Creating DIRECTED-ONLY regime performance visualization (DPAH + Twitter) for supplementary materials...")
    
    # Load directed-only algorithm data
    algorithm_data = prepare_directed_algorithm_data()
    
    if not algorithm_data:
        print("No data found for directed networks. Please check regime_based_metrics file.")
        return
    
    print(f"Loaded directed network data for regimes: {list(algorithm_data.keys())}")
    
    # Create output directory
    output_dir = "../../Figs/Regime_Analysis"
    os.makedirs(output_dir, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    
    # Generate comprehensive three-panel plot for directed networks
    print("Creating comprehensive three-panel plot for directed networks...")
    fig1 = create_three_panel_plot(algorithm_data)
    output_path1 = f"{output_dir}/regime_performance_directed_only_{today}.pdf"
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"Directed-only comprehensive plot saved: {output_path1}")
    plt.show()
    plt.close()
    
    print("Directed-only regime visualization complete!")
    
    # Print key insights for directed networks
    print("\n=== KEY INSIGHTS (DIRECTED NETWORKS ONLY) ===")
    for regime in ['low', 'medium', 'high']:
        if regime in algorithm_data:
            algorithms = algorithm_data[regime]['algorithms']
            cooperation = algorithm_data[regime]['cooperation']
            
            # Find best and worst performers
            best_idx = np.argmax(cooperation)
            worst_idx = np.argmin(cooperation)
            
            print(f"{regime.capitalize()} regime:")
            print(f"  Best: {algorithms[best_idx]} ({cooperation[best_idx]:.3f})")
            print(f"  Worst: {algorithms[worst_idx]} ({cooperation[worst_idx]:.3f})")
            
            # Find WTF performance if available
            wtf_scenarios = [alg for alg in algorithms if 'wtf' in alg.lower()]
            if wtf_scenarios:
                wtf_idx = algorithms.index(wtf_scenarios[0])
                wtf_rank = sorted(range(len(cooperation)), key=lambda i: cooperation[i], reverse=True).index(wtf_idx) + 1
                print(f"  WTF: {algorithms[wtf_idx]} ({cooperation[wtf_idx]:.3f}) - Rank {wtf_rank}/{len(algorithms)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--directed-only":
        main_directed_only()
    else:
        main()