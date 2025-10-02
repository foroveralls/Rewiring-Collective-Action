#!/usr/bin/env python3
"""
Consolidated continuous regime analysis visualizations
Combines the best features from novel_regime_visualizations_fixed.py and continuous_phase_diagrams.py
All visualizations use averaged data per stubbornness parameter value (not all parameter combinations)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from datetime import date
from scipy.interpolate import griddata, interp1d
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import glob

cm = 1/2.54
FONT_SIZE = 6

# Color scheme matching existing plots
FRIENDLY_COLORS = {
    'static': '#EE7733', 'random': '#0077BB', 'local (similar)': '#AA4499', 
    'local (opposite)': '#117733', 'bridge (similar)': '#CC3311', 'bridge (opposite)': '#EE3377',
    'empirical wtf': '#BBBBBB', 'empirical node2vec': '#44BB99',
    'Static': '#EE7733', 'Random': '#0077BB', 'Similar': '#AA4499', 
    'Opposite': '#117733', 'Bridge (Similar)': '#CC3311', 'Bridge (Opposite)': '#EE3377',
    'WTF': '#BBBBBB', 'Node2Vec': '#44BB99'
}

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

def load_continuous_data():
    """Load continuous parameter data and average per stubbornness value"""
    # Find heatmap files with stubbornness parameter
    heatmap_files = glob.glob("../../Output/heatmap_sweep_*stubbornness*.csv")

    if not heatmap_files:
        print("No heatmap files with stubbornness parameter found in Output directory")
        return None

    # Use the most recent file
    file_path = sorted(heatmap_files)[-1]
    print(f"Loading data from: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records from continuous data")

    # Check if required columns exist
    if 'stubbornness' not in df.columns:
        print(f"Error: stubbornness column not found. Available columns: {list(df.columns)}")
        return None

    # Create scenario column from rewiring and mode
    df['rewiring'] = df['rewiring'].fillna('none')
    df['mode'] = df['mode'].fillna('none')
    df['scenario'] = df['rewiring'] + ' ' + df['mode']

    # Create cleaner algorithm names
    algorithm_mapping = {
        'diff biased': 'Opposite',
        'same biased': 'Similar',
        'wtf empirical': 'WTF',
        'node2vec empirical': 'Node2Vec',
        'none none': 'Static',
        'none random': 'Random',
        'diff bridge': 'Bridge (Opposite)',
        'same bridge': 'Bridge (Similar)',
        'none wtf': 'WTF',
        'none node2vec': 'Node2Vec'
    }
    df['algorithm'] = df['scenario'].map(algorithm_mapping).fillna(df['scenario'])

    # Group by stubbornness value and algorithm, calculate averages across all runs
    # This gives us one averaged data point per (stubbornness, algorithm) combination
    aggregated_data = []

    for (stub_val, algorithm), group in df.groupby(['stubbornness', 'algorithm']):
        # Calculate cooperative states (state > 0)
        cooperative_mask = group['state'] > 0
        n_cooperative = cooperative_mask.sum()
        n_total = len(group)

        metrics = {
            'stubbornness': stub_val,
            'stubbornness_numeric': stub_val,  # Keep for backward compatibility
            'algorithm': algorithm,
            'mean_cooperation': group['state'].mean(),
            'mean_polarization': group['state_std'].mean(),
            'cooperative_volume_percent': (n_cooperative / n_total) * 100 if n_total > 0 else 0.0,
            'n_runs': n_total
        }
        aggregated_data.append(metrics)

    result_df = pd.DataFrame(aggregated_data)
    print(f"Aggregated to {len(result_df)} (stubbornness, algorithm) combinations")
    print(f"Algorithms: {sorted(result_df['algorithm'].unique())}")
    print(f"Stubbornness values: {sorted(result_df['stubbornness'].unique())}")

    return result_df

def load_2d_parameter_data():
    """Load continuous parameter data preserving both stubbornness and backfirer fraction dimensions"""
    # Find heatmap files with stubbornness parameter
    heatmap_files = glob.glob("../../Output/heatmap_sweep_*stubbornness*.csv")

    if not heatmap_files:
        print("No heatmap files with stubbornness parameter found in Output directory")
        return None

    # Use the most recent file
    file_path = sorted(heatmap_files)[-1]
    print(f"Loading 2D parameter data from: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records")

    # Check if required columns exist
    required_cols = ['stubbornness', 'polarisingNode_f']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Required columns not found. Available: {list(df.columns)}")
        return None

    # Create scenario column from rewiring and mode
    df['rewiring'] = df['rewiring'].fillna('none')
    df['mode'] = df['mode'].fillna('none')
    df['scenario'] = df['rewiring'] + ' ' + df['mode']

    # Create cleaner algorithm names
    algorithm_mapping = {
        'diff biased': 'Opposite',
        'same biased': 'Similar',
        'wtf empirical': 'WTF',
        'node2vec empirical': 'Node2Vec',
        'none none': 'Static',
        'none random': 'Random',
        'diff bridge': 'Bridge (Opposite)',
        'same bridge': 'Bridge (Similar)',
        'none wtf': 'WTF',
        'none node2vec': 'Node2Vec'
    }
    df['algorithm'] = df['scenario'].map(algorithm_mapping).fillna(df['scenario'])

    # Group by stubbornness, backfirer fraction, and algorithm
    aggregated_data = []

    for (stub_val, bf_val, algorithm), group in df.groupby(['stubbornness', 'polarisingNode_f', 'algorithm']):
        # Calculate cooperative states (state > 0)
        cooperative_mask = group['state'] > 0
        n_cooperative = cooperative_mask.sum()
        n_total = len(group)

        metrics = {
            'stubbornness': stub_val,
            'backfirer_fraction': bf_val,
            'algorithm': algorithm,
            'mean_cooperation': group['state'].mean(),
            'mean_polarization': group['state_std'].mean(),
            'cooperative_volume_percent': (n_cooperative / n_total) * 100 if n_total > 0 else 0.0,
            'n_runs': n_total
        }
        aggregated_data.append(metrics)

    result_df = pd.DataFrame(aggregated_data)
    print(f"Aggregated to {len(result_df)} (stubbornness, backfirer_fraction, algorithm) combinations")
    print(f"Algorithms: {sorted(result_df['algorithm'].unique())}")
    print(f"Stubbornness values: {len(result_df['stubbornness'].unique())} unique")
    print(f"Backfirer fraction values: {len(result_df['backfirer_fraction'].unique())} unique")

    return result_df

def create_2d_phase_diagram(df):
    """Create 2D phase diagram showing cooperation vs polarization with stubbornness color-coded"""
    setup_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(10*cm, 8*cm))
    
    # Focus on main algorithms for clarity
    main_algorithms = ['Opposite', 'Similar', 'WTF', 'Node2Vec', 'Static', 'Random']
    df_main = df[df['algorithm'].isin(main_algorithms)]
    
    # Create algorithm color mapping
    algorithm_colors = {
        'Opposite': FRIENDLY_COLORS['local (opposite)'],
        'Similar': FRIENDLY_COLORS['local (similar)'],
        'WTF': FRIENDLY_COLORS['empirical wtf'],
        'Node2Vec': FRIENDLY_COLORS['empirical node2vec'],
        'Static': FRIENDLY_COLORS['static'],
        'Random': FRIENDLY_COLORS['random']
    }
    
    # Create scatter plot with stubbornness as color intensity
    for alg in main_algorithms:
        alg_data = df_main[df_main['algorithm'] == alg]
        if len(alg_data) > 0:
            # Use stubbornness_numeric to vary color intensity
            colors = []
            base_color = algorithm_colors.get(alg, '#666666')
            
            for _, row in alg_data.iterrows():
                stub_val = row['stubbornness_numeric']
                # Create gradient from light to dark based on stubbornness
                if stub_val <= 0.3:  # Low stubbornness
                    alpha = 0.4
                    size = 20
                    marker = 'o'
                elif stub_val <= 0.6:  # Medium stubbornness
                    alpha = 0.7
                    size = 25
                    marker = 's'
                else:  # High stubbornness
                    alpha = 1.0
                    size = 30
                    marker = '^'
                
                ax.scatter(row['mean_cooperation'], row['mean_polarization'],
                          c=base_color, alpha=alpha, s=size, marker=marker,
                          edgecolors='white', linewidth=0.3)
    
    # Draw trajectory lines connecting same algorithm across stubbornness levels
    for alg in main_algorithms:
        alg_data = df_main[df_main['algorithm'] == alg].sort_values('stubbornness_numeric')
        if len(alg_data) >= 2:
            ax.plot(alg_data['mean_cooperation'], alg_data['mean_polarization'],
                   color=algorithm_colors.get(alg, '#666666'), 
                   linewidth=0.8, alpha=0.6, linestyle='--')
    
    # Add regime boundaries as background regions
    coop_range = ax.get_xlim()
    polar_range = ax.get_ylim()
    
    # Add subtle background regions to show emergent regimes
    # Low stubbornness region (high polarization)
    rect1 = patches.Rectangle((coop_range[0], 0.75), coop_range[1]-coop_range[0], 
                             polar_range[1]-0.75, linewidth=0, 
                             facecolor='lightblue', alpha=0.1, label='Low stubbornness\n(high polarization)')
    ax.add_patch(rect1)
    
    # Medium stubbornness region
    rect2 = patches.Rectangle((coop_range[0], 0.6), coop_range[1]-coop_range[0], 
                             0.15, linewidth=0, 
                             facecolor='lightyellow', alpha=0.1, label='Medium stubbornness')
    ax.add_patch(rect2)
    
    # High stubbornness region (low polarization)
    rect3 = patches.Rectangle((coop_range[0], polar_range[0]), coop_range[1]-coop_range[0], 
                             0.6-polar_range[0], linewidth=0, 
                             facecolor='lightcoral', alpha=0.1, label='High stubbornness\n(low polarization)')
    ax.add_patch(rect3)
    
    # Customize axes
    ax.set_xlabel('Cooperation ($a$)', fontsize=FONT_SIZE-1)
    ax.set_ylabel('Polarization ($\sigma(a)$)', fontsize=FONT_SIZE-1)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.tick_params(labelsize=FONT_SIZE-2)
    
    # Create custom legends
    from matplotlib.lines import Line2D
    
    # Algorithm legend
    alg_legend = [Line2D([0], [0], marker='o', color='w', 
                        markerfacecolor=algorithm_colors.get(alg, '#666666'),
                        markersize=4, label=alg, alpha=0.8) 
                 for alg in main_algorithms]
    
    # Stubbornness legend
    stub_legend = [
        Line2D([0], [0], marker='o', color='gray', markersize=4, 
               label='Low stubbornness', alpha=0.4, linestyle='None'),
        Line2D([0], [0], marker='s', color='gray', markersize=5, 
               label='Medium stubbornness', alpha=0.7, linestyle='None'),
        Line2D([0], [0], marker='^', color='gray', markersize=6, 
               label='High stubbornness', alpha=1.0, linestyle='None')
    ]
    
    # Add legends
    leg1 = ax.legend(handles=alg_legend, loc='upper left', 
                    fontsize=FONT_SIZE-2, title='Algorithm')
    leg2 = ax.legend(handles=stub_legend, loc='lower right', 
                    fontsize=FONT_SIZE-2, title='Stubbornness Level')
    ax.add_artist(leg1)
    
    plt.tight_layout()
    return fig

def create_performance_landscape_plot(df):
    """Create 2D performance landscape visualization with averaged stubbornness data"""
    setup_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(12*cm, 10*cm))
    
    # Focus on main algorithms for clarity
    main_algorithms = ['Opposite', 'Similar', 'WTF', 'Node2Vec', 'Static', 'Random']
    df_main = df[df['algorithm'].isin(main_algorithms)]
    
    if len(df_main) == 0:
        print(f"No data found for main algorithms. Available: {sorted(df['algorithm'].unique())}")
        return fig
    
    print(f"Data points per algorithm:")
    for alg in main_algorithms:
        alg_data = df_main[df_main['algorithm'] == alg]
        if len(alg_data) > 0:
            print(f"  {alg}: {len(alg_data)} points")
            print(f"    Stubbornness range: {alg_data['stubbornness'].min():.2f} - {alg_data['stubbornness'].max():.2f}")
            print(f"    Cooperation range: {alg_data['mean_cooperation'].min():.2f} - {alg_data['mean_cooperation'].max():.2f}")
    
    # Create scatter plot with averaged data points
    for alg in main_algorithms:
        alg_data = df_main[df_main['algorithm'] == alg]
        if len(alg_data) > 0:
            color = FRIENDLY_COLORS.get(alg, '#666666')
            
            # Scatter plot with size based on cooperative volume
            scatter = ax.scatter(alg_data['mean_cooperation'], alg_data['mean_polarization'],
                               c=alg_data['stubbornness'], cmap='viridis',
                               s=alg_data['cooperative_volume_percent']/3,  # Reduced size scaling
                               alpha=0.7, edgecolors='white', linewidth=0.3,  # Thinner edges
                               label=alg)
            
            # Draw trajectory connecting points sorted by stubbornness
            alg_data_sorted = alg_data.sort_values('stubbornness')
            if len(alg_data_sorted) >= 2:
                ax.plot(alg_data_sorted['mean_cooperation'], alg_data_sorted['mean_polarization'],
                       color=color, linewidth=0.8, alpha=0.6, linestyle='--')  # Thinner lines
    
    # Add colorbar for stubbornness
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Stubbornness Parameter', fontsize=FONT_SIZE-1)
    cbar.ax.tick_params(labelsize=FONT_SIZE-2)
    
    # Customize axes
    ax.set_xlabel('Cooperation ($a$)', fontsize=FONT_SIZE-1)
    ax.set_ylabel('Polarization ($\sigma(a)$)', fontsize=FONT_SIZE-1)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.tick_params(labelsize=FONT_SIZE-2)
    
    # Legend for algorithms (outside plot area)
    legend = ax.legend(bbox_to_anchor=(1.25, 1), loc='upper left', 
                      fontsize=FONT_SIZE-2, title='Algorithm', framealpha=0.9)
    legend.get_title().set_fontsize(FONT_SIZE-1)
    
    # Add text explaining point size
    ax.text(0.02, 0.98, 'Point size ∝ Cooperative Volume', 
           transform=ax.transAxes, fontsize=FONT_SIZE-3, 
           verticalalignment='top', alpha=0.7,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_continuous_landscape_plot(df):
    """Create continuous landscape plot with cooperative volume as elevation"""
    setup_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(10*cm, 8*cm))
    
    # Use all data points for landscape generation
    coop_vals = df['mean_cooperation'].values
    polar_vals = df['mean_polarization'].values
    volume_vals = df['cooperative_volume_percent'].values
    stub_vals = df['stubbornness_numeric'].values
    
    # Create high-resolution grid for smooth landscape
    coop_range = np.linspace(coop_vals.min(), coop_vals.max(), 100)
    polar_range = np.linspace(polar_vals.min(), polar_vals.max(), 100)
    coop_grid, polar_grid = np.meshgrid(coop_range, polar_range)
    
    # Interpolate cooperative volume for landscape
    volume_grid = griddata((coop_vals, polar_vals), volume_vals, 
                          (coop_grid, polar_grid), method='cubic')
    
    # Interpolate stubbornness for color overlay
    stub_grid = griddata((coop_vals, polar_vals), stub_vals, 
                        (coop_grid, polar_grid), method='cubic')
    
    # Create filled contour for cooperative volume (landscape elevation)
    contourf = ax.contourf(coop_grid, polar_grid, volume_grid, levels=20, 
                          cmap='RdYlBu_r', alpha=0.6)
    
    # Add contour lines for cooperative volume
    contours = ax.contour(coop_grid, polar_grid, volume_grid, levels=15, 
                         colors='white', alpha=0.8, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=FONT_SIZE-3, fmt='%1.0f%%', 
             colors='white')
    
    # Overlay stubbornness information with second contour set
    stub_contours = ax.contour(coop_grid, polar_grid, stub_grid, 
                              levels=[0.3, 0.6], colors='black', 
                              linewidths=1, linestyles=['--', ':'])
    ax.clabel(stub_contours, inline=True, fontsize=FONT_SIZE-2, 
             fmt={0.3: 'Low/Med', 0.6: 'Med/High'}, colors='black')
    
    # Plot all data points with algorithm-specific colors
    main_algorithms = ['Opposite', 'Similar', 'WTF', 'Node2Vec', 'Static', 'Random']
    algorithm_colors = {
        'Opposite': FRIENDLY_COLORS['local (opposite)'],
        'Similar': FRIENDLY_COLORS['local (similar)'],
        'WTF': FRIENDLY_COLORS['empirical wtf'],
        'Node2Vec': FRIENDLY_COLORS['empirical node2vec'],
        'Static': FRIENDLY_COLORS['static'],
        'Random': FRIENDLY_COLORS['random']
    }
    
    # Plot points with varying size based on stubbornness
    for alg in main_algorithms:
        alg_data = df[df['algorithm'] == alg]
        if len(alg_data) > 0:
            for _, row in alg_data.iterrows():
                stub_val = row['stubbornness_numeric']
                size = 15 + stub_val * 25  # Size increases with stubbornness
                
                ax.scatter(row['mean_cooperation'], row['mean_polarization'],
                          c=algorithm_colors.get(alg, '#666666'), 
                          s=size, alpha=0.9, edgecolors='white', linewidth=0.5)
            
            # Draw trajectory
            alg_data_sorted = alg_data.sort_values('stubbornness_numeric')
            if len(alg_data_sorted) >= 2:
                ax.plot(alg_data_sorted['mean_cooperation'], 
                       alg_data_sorted['mean_polarization'],
                       color=algorithm_colors.get(alg, '#666666'), 
                       linewidth=1.5, alpha=0.8)
    
    # Color bar for landscape
    cbar = plt.colorbar(contourf, ax=ax, shrink=0.8)
    cbar.set_label('Cooperative Volume (%)', fontsize=FONT_SIZE-1)
    cbar.ax.tick_params(labelsize=FONT_SIZE-2)
    
    # Customize axes
    ax.set_xlabel('Cooperation ($a$)', fontsize=FONT_SIZE-1)
    ax.set_ylabel('Polarization ($\sigma(a)$)', fontsize=FONT_SIZE-1)
    ax.grid(True, alpha=0.2, linewidth=0.3)
    ax.tick_params(labelsize=FONT_SIZE-2)
    
    # Legend for algorithms
    from matplotlib.lines import Line2D
    alg_legend = [Line2D([0], [0], marker='o', color='w', 
                        markerfacecolor=algorithm_colors.get(alg, '#666666'),
                        markersize=5, label=alg, alpha=0.9) 
                 for alg in main_algorithms]
    
    ax.legend(handles=alg_legend, loc='upper left', 
             fontsize=FONT_SIZE-2, title='Algorithm', framealpha=0.9)
    
    plt.tight_layout()
    return fig

def create_stubbornness_trajectory_plot(df):
    """Create plot showing how cooperation and polarization change with stubbornness"""
    setup_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10*cm, 12*cm), sharex=True)
    
    # Focus on main algorithms
    main_algorithms = ['Opposite', 'Similar', 'WTF', 'Node2Vec', 'Static', 'Random']
    df_main = df[df['algorithm'].isin(main_algorithms)]
    
    # Plot cooperation vs stubbornness
    for alg in main_algorithms:
        alg_data = df_main[df_main['algorithm'] == alg].sort_values('stubbornness')
        if len(alg_data) > 0:
            color = FRIENDLY_COLORS.get(alg, '#666666')
            
            # Data is already averaged per stubbornness value, no need to re-group
            # Plot cooperation
            ax1.plot(alg_data['stubbornness'], alg_data['mean_cooperation'],
                    'o-', color=color, linewidth=1, markersize=2.5, 
                    alpha=0.8, label=alg)
            
            # Plot polarization  
            ax2.plot(alg_data['stubbornness'], alg_data['mean_polarization'],
                    'o-', color=color, linewidth=1, markersize=2.5, 
                    alpha=0.8, label=alg)
    
    # Customize cooperation plot
    ax1.set_ylabel('Cooperation ($a$)', fontsize=FONT_SIZE-1)
    ax1.grid(True, alpha=0.3, linewidth=0.3)
    ax1.tick_params(labelsize=FONT_SIZE-2)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              fontsize=FONT_SIZE-2, title='Algorithm')
    
    # Customize polarization plot
    ax2.set_xlabel('Stubbornness Parameter', fontsize=FONT_SIZE-1)
    ax2.set_ylabel('Polarization ($\sigma(a)$)', fontsize=FONT_SIZE-1)
    ax2.grid(True, alpha=0.3, linewidth=0.3)
    ax2.tick_params(labelsize=FONT_SIZE-2)
    
    plt.tight_layout()
    return fig

def create_continuous_single_panel_plot(df):
    """Create single-panel plot matching stubbornness_trajectories.pdf data structure"""
    setup_style()
    
    # Create figure with single panel and dual y-axes
    fig, ax1 = plt.subplots(1, 1, figsize=(8*cm, 6*cm))
    
    # Filter main algorithms
    main_algorithms = ['Opposite', 'Similar', 'WTF', 'Node2Vec', 'Static', 'Random']
    df_main = df[df['algorithm'].isin(main_algorithms)]
    
    algorithm_colors = {
        'Opposite': FRIENDLY_COLORS['local (opposite)'],
        'Similar': FRIENDLY_COLORS['local (similar)'],
        'WTF': FRIENDLY_COLORS['empirical wtf'],
        'Node2Vec': FRIENDLY_COLORS['empirical node2vec'],
        'Static': FRIENDLY_COLORS['static'],
        'Random': FRIENDLY_COLORS['random']
    }
    
    # Create twin axis for polarization
    ax2 = ax1.twinx()
    
    # Plot cooperation and polarization for each algorithm
    for alg in main_algorithms:
        alg_data = df_main[df_main['algorithm'] == alg]
        if len(alg_data) > 0:
            color = algorithm_colors.get(alg, '#666666')
            
            # Sort by stubbornness for smooth lines
            alg_data_sorted = alg_data.sort_values('stubbornness_numeric')
            
            # Plot cooperation (solid line, left y-axis)
            ax1.plot(alg_data_sorted['stubbornness_numeric'], 
                    alg_data_sorted['mean_cooperation'],
                    'o-', color=color, linewidth=1, markersize=2.5, 
                    alpha=0.8, label=alg)
            
            # Plot polarization (dotted line, right y-axis)
            ax2.plot(alg_data_sorted['stubbornness_numeric'], 
                    alg_data_sorted['mean_polarization'],
                    'o:', color=color, linewidth=1, markersize=2.5, 
                    alpha=0.6)
    
    # Customize left axis (cooperation)
    ax1.set_xlabel('Stubbornness Parameter', fontsize=FONT_SIZE-1, labelpad=2)
    ax1.set_ylabel('$a$', fontsize=FONT_SIZE-1, labelpad=2, color='black')
    ax1.grid(True, alpha=0.3, linewidth=0.3)
    ax1.tick_params(labelsize=FONT_SIZE-2)
    
    # Customize right axis (polarization)
    ax2.set_ylabel('$\sigma(a)$', fontsize=FONT_SIZE-1, labelpad=2, color='black')
    ax2.tick_params(labelsize=FONT_SIZE-2)
    
    # Add regime boundary lines (but don't label them as regimes)
    ax1.axvline(0.4, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax1.axvline(0.7, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    
    # Add subtle regime labels at top
    ax1.text(0.2, ax1.get_ylim()[1]*0.98, 'Low', ha='center', 
            fontsize=FONT_SIZE-3, color='gray', alpha=0.7)
    ax1.text(0.55, ax1.get_ylim()[1]*0.98, 'Medium', ha='center', 
            fontsize=FONT_SIZE-3, color='gray', alpha=0.7)
    ax1.text(0.85, ax1.get_ylim()[1]*0.98, 'High', ha='center', 
            fontsize=FONT_SIZE-3, color='gray', alpha=0.7)
    
    # Legend at bottom
    fig.legend(bbox_to_anchor=(0.5, 0.02), loc='upper center', ncol=3,
              fontsize=FONT_SIZE-2, frameon=True, fancybox=False, 
              edgecolor='black', facecolor='white')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for legend
    
    return fig

def create_stubbornness_parameter_space(df):
    """Create parameter space plot showing stubbornness as continuous variable"""
    setup_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12*cm, 6*cm))

    # Filter main algorithms
    main_algorithms = ['Opposite', 'Similar', 'WTF', 'Node2Vec', 'Static', 'Random']
    df_main = df[df['algorithm'].isin(main_algorithms)]

    algorithm_colors = {
        'Opposite': FRIENDLY_COLORS['local (opposite)'],
        'Similar': FRIENDLY_COLORS['local (similar)'],
        'WTF': FRIENDLY_COLORS['empirical wtf'],
        'Node2Vec': FRIENDLY_COLORS['empirical node2vec'],
        'Static': FRIENDLY_COLORS['static'],
        'Random': FRIENDLY_COLORS['random']
    }

    # Panel 1: Stubbornness vs Cooperation
    for alg in main_algorithms:
        alg_data = df_main[df_main['algorithm'] == alg]
        if len(alg_data) > 0:
            color = algorithm_colors.get(alg, '#666666')
            ax1.scatter(alg_data['stubbornness_numeric'], alg_data['mean_cooperation'],
                       color=color, alpha=0.8, s=25, label=alg)
            # Fit polynomial trend line
            if len(alg_data) >= 2:
                stub_sorted = alg_data.sort_values('stubbornness_numeric')
                ax1.plot(stub_sorted['stubbornness_numeric'],
                        stub_sorted['mean_cooperation'],
                        color=color, linewidth=1, alpha=0.7)

    ax1.set_xlabel('Stubbornness Parameter', fontsize=FONT_SIZE-1)
    ax1.set_ylabel('Cooperation ($a$)', fontsize=FONT_SIZE-1)
    ax1.grid(True, alpha=0.3, linewidth=0.3)
    ax1.tick_params(labelsize=FONT_SIZE-2)
    ax1.set_title('A', fontsize=FONT_SIZE, fontweight='bold')

    # Add regime boundaries as vertical lines
    ax1.axvline(0.4, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(0.7, color='gray', linestyle='--', alpha=0.5)
    ax1.text(0.2, ax1.get_ylim()[1], 'Low', ha='center', fontsize=FONT_SIZE-2, color='gray')
    ax1.text(0.55, ax1.get_ylim()[1], 'Medium', ha='center', fontsize=FONT_SIZE-2, color='gray')
    ax1.text(0.85, ax1.get_ylim()[1], 'High', ha='center', fontsize=FONT_SIZE-2, color='gray')

    # Panel 2: Stubbornness vs Polarization
    for alg in main_algorithms:
        alg_data = df_main[df_main['algorithm'] == alg]
        if len(alg_data) > 0:
            color = algorithm_colors.get(alg, '#666666')
            ax2.scatter(alg_data['stubbornness_numeric'], alg_data['mean_polarization'],
                       color=color, alpha=0.8, s=25)
            if len(alg_data) >= 2:
                stub_sorted = alg_data.sort_values('stubbornness_numeric')
                ax2.plot(stub_sorted['stubbornness_numeric'],
                        stub_sorted['mean_polarization'],
                        color=color, linewidth=1, alpha=0.7)

    ax2.set_xlabel('Stubbornness Parameter', fontsize=FONT_SIZE-1)
    ax2.set_ylabel('Polarization ($\sigma(a)$)', fontsize=FONT_SIZE-1)
    ax2.grid(True, alpha=0.3, linewidth=0.3)
    ax2.tick_params(labelsize=FONT_SIZE-2)
    ax2.set_title('B', fontsize=FONT_SIZE, fontweight='bold')

    # Add regime boundaries
    ax2.axvline(0.4, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(0.7, color='gray', linestyle='--', alpha=0.5)

    # Shared legend
    fig.legend(bbox_to_anchor=(0.5, 0.02), loc='upper center', ncol=3,
              fontsize=FONT_SIZE-2, frameon=True, fancybox=False,
              edgecolor='black', facecolor='white')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    return fig

def create_2d_heatmap_grid(df_2d):
    """Create 2D heatmap grid showing cooperation and polarization as function of stubbornness and backfirer fraction"""
    setup_style()

    # Filter main algorithms
    main_algorithms = ['Opposite', 'Similar', 'WTF', 'Node2Vec', 'Static', 'Random']

    # Create figure with subplots: 6 algorithms × 2 metrics (cooperation, polarization)
    fig, axes = plt.subplots(6, 2, figsize=(12*cm, 24*cm))

    for idx, alg in enumerate(main_algorithms):
        alg_data = df_2d[df_2d['algorithm'] == alg]

        if len(alg_data) == 0:
            print(f"Warning: No data for {alg}")
            continue

        # Create pivot tables for heatmaps
        coop_pivot = alg_data.pivot_table(
            values='mean_cooperation',
            index='backfirer_fraction',
            columns='stubbornness',
            aggfunc='mean'
        )

        polar_pivot = alg_data.pivot_table(
            values='mean_polarization',
            index='backfirer_fraction',
            columns='stubbornness',
            aggfunc='mean'
        )

        # Plot cooperation heatmap (left column)
        ax_coop = axes[idx, 0]
        im1 = ax_coop.imshow(coop_pivot, aspect='auto', origin='lower',
                            cmap='RdBu_r', vmin=-1, vmax=1, interpolation='bilinear')
        ax_coop.set_title(f'{alg}', fontsize=FONT_SIZE-1, fontweight='bold')

        # Set ticks
        n_stub = len(coop_pivot.columns)
        n_bf = len(coop_pivot.index)
        ax_coop.set_xticks(np.linspace(0, n_stub-1, 5))
        ax_coop.set_xticklabels([f'{x:.1f}' for x in np.linspace(0, 1, 5)], fontsize=FONT_SIZE-2)
        ax_coop.set_yticks(np.linspace(0, n_bf-1, 5))
        ax_coop.set_yticklabels([f'{x:.1f}' for x in np.linspace(0, 1, 5)], fontsize=FONT_SIZE-2)

        if idx == 5:  # Bottom row
            ax_coop.set_xlabel('Stubbornness', fontsize=FONT_SIZE-1)
        else:
            ax_coop.set_xticklabels([])

        if idx == 0:  # Top row
            ax_coop.text(0.5, 1.15, 'Cooperation ($a$)', transform=ax_coop.transAxes,
                        ha='center', fontsize=FONT_SIZE, fontweight='bold')

        ax_coop.set_ylabel('Backfirer Fraction', fontsize=FONT_SIZE-2)

        # Plot polarization heatmap (right column)
        ax_polar = axes[idx, 1]
        im2 = ax_polar.imshow(polar_pivot, aspect='auto', origin='lower',
                             cmap='YlOrRd', vmin=0, vmax=1, interpolation='bilinear')

        # Set ticks
        ax_polar.set_xticks(np.linspace(0, n_stub-1, 5))
        ax_polar.set_xticklabels([f'{x:.1f}' for x in np.linspace(0, 1, 5)], fontsize=FONT_SIZE-2)
        ax_polar.set_yticks(np.linspace(0, n_bf-1, 5))
        ax_polar.set_yticklabels([f'{x:.1f}' for x in np.linspace(0, 1, 5)], fontsize=FONT_SIZE-2)

        if idx == 5:  # Bottom row
            ax_polar.set_xlabel('Stubbornness', fontsize=FONT_SIZE-1)
        else:
            ax_polar.set_xticklabels([])

        if idx == 0:  # Top row
            ax_polar.text(0.5, 1.15, 'Polarization ($\sigma(a)$)', transform=ax_polar.transAxes,
                         ha='center', fontsize=FONT_SIZE, fontweight='bold')

        ax_polar.set_ylabel('')
        ax_polar.set_yticklabels([])

    # Add colorbars at the bottom
    cbar1 = fig.colorbar(im1, ax=axes[:, 0], orientation='horizontal',
                         pad=0.08, aspect=40, shrink=0.8)
    cbar1.set_label('Cooperation', fontsize=FONT_SIZE-1)
    cbar1.ax.tick_params(labelsize=FONT_SIZE-2)

    cbar2 = fig.colorbar(im2, ax=axes[:, 1], orientation='horizontal',
                         pad=0.08, aspect=40, shrink=0.8)
    cbar2.set_label('Polarization', fontsize=FONT_SIZE-1)
    cbar2.ax.tick_params(labelsize=FONT_SIZE-2)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.15)
    return fig

def main():
    """Generate consolidated continuous regime analysis visualizations"""
    print("Creating consolidated continuous regime analysis visualizations...")
    
    # Load continuous data
    df = load_continuous_data()
    if df is None:
        return
    
    print(f"Working with {len(df)} data points")
    print(f"Algorithms found: {sorted(df['algorithm'].unique())}")
    print(f"Stubbornness range: {df['stubbornness_numeric'].min():.3f} - {df['stubbornness_numeric'].max():.3f}")
    
    # Create output directory
    output_dir = "../../Figs/Regime_Analysis"
    os.makedirs(output_dir, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    
    # Generate 2D phase diagram
    print("Creating 2D phase diagram...")
    fig1 = create_2d_phase_diagram(df)
    output_path1 = f"{output_dir}/continuous_phase_diagram_{today}.pdf"
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"Phase diagram saved: {output_path1}")
    plt.show()
    plt.close()
    
    # Generate performance landscape
    print("Creating performance landscape...")
    fig2 = create_performance_landscape_plot(df)
    output_path2 = f"{output_dir}/continuous_performance_landscape_{today}.pdf"
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Performance landscape saved: {output_path2}")
    plt.show()
    plt.close()
    
    # Generate continuous landscape with elevation
    print("Creating continuous landscape with elevation...")
    fig3 = create_continuous_landscape_plot(df)
    output_path3 = f"{output_dir}/continuous_landscape_elevation_{today}.pdf"
    fig3.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"Continuous landscape saved: {output_path3}")
    plt.show()
    plt.close()
    
    # Generate stubbornness trajectory plot
    print("Creating stubbornness trajectory plot...")
    fig4 = create_stubbornness_trajectory_plot(df)
    output_path4 = f"{output_dir}/stubbornness_trajectories_{today}.pdf"
    fig4.savefig(output_path4, dpi=300, bbox_inches='tight')
    print(f"Trajectory plot saved: {output_path4}")
    plt.show()
    plt.close()
    
    # Generate continuous single panel plot
    print("Creating continuous single panel plot...")
    fig5 = create_continuous_single_panel_plot(df)
    output_path5 = f"{output_dir}/continuous_single_panel_{today}.pdf"
    fig5.savefig(output_path5, dpi=300, bbox_inches='tight')
    print(f"Single panel plot saved: {output_path5}")
    plt.show()
    plt.close()
    
    # Generate stubbornness parameter space
    print("Creating stubbornness parameter space...")
    fig6 = create_stubbornness_parameter_space(df)
    output_path6 = f"{output_dir}/stubbornness_parameter_space_{today}.pdf"
    fig6.savefig(output_path6, dpi=300, bbox_inches='tight')
    print(f"Parameter space plot saved: {output_path6}")
    plt.show()
    plt.close()
    
    print("Consolidated continuous visualization generation complete!")

if __name__ == "__main__":
    main()