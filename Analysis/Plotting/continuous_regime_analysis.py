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
FONT_SIZE = 8

# Analysis parameters
MAX_BACKFIRER_FRACTION = 1.0  # Filter data by backfirer fraction (default: include all)

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

def load_continuous_data(max_backfirer_fraction=None):
    """Load continuous parameter data and average per stubbornness value

    Args:
        max_backfirer_fraction: Maximum backfirer fraction to include (default: None uses global MAX_BACKFIRER_FRACTION)

    Filters to backfirer_fraction <= max_backfirer_fraction.
    Also calculates min/max ranges across backfirer fractions for uncertainty visualization.
    """
    # Use parameter if provided, otherwise use global constant
    if max_backfirer_fraction is None:
        max_backfirer_fraction = MAX_BACKFIRER_FRACTION
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

    # Filter for backfirer_fraction <= max_backfirer_fraction
    if 'polarisingNode_f' in df.columns:
        initial_count = len(df)
        df = df[df['polarisingNode_f'] <= max_backfirer_fraction]
        print(f"Filtered to backfirer_fraction <= {max_backfirer_fraction}: {len(df)} records (removed {initial_count - len(df)})")
    else:
        print("Warning: polarisingNode_f column not found, cannot filter by backfirer fraction")

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

    # Group by stubbornness value and algorithm, calculate averages AND ranges
    # This gives us one averaged data point per (stubbornness, algorithm) combination
    aggregated_data = []

    for (stub_val, algorithm), group in df.groupby(['stubbornness', 'algorithm']):
        # Calculate cooperative states (state > 0)
        cooperative_mask = group['state'] > 0
        n_cooperative = cooperative_mask.sum()
        n_total = len(group)

        # For uncertainty bands: calculate min/max cooperation across different backfirer fractions
        # Group by backfirer fraction first to get one value per backfirer level
        if 'polarisingNode_f' in group.columns:
            bf_group_coop = group.groupby('polarisingNode_f')['state'].mean()
            bf_group_polar = group.groupby('polarisingNode_f')['state_std'].mean()
            min_coop = bf_group_coop.min()
            max_coop = bf_group_coop.max()
            min_polar = bf_group_polar.min()
            max_polar = bf_group_polar.max()
        else:
            min_coop = group['state'].min()
            max_coop = group['state'].max()
            min_polar = group['state_std'].min()
            max_polar = group['state_std'].max()

        metrics = {
            'stubbornness': stub_val,
            'stubbornness_numeric': stub_val,  # Keep for backward compatibility
            'algorithm': algorithm,
            'mean_cooperation': group['state'].mean(),
            'min_cooperation': min_coop,
            'max_cooperation': max_coop,
            'mean_polarization': group['state_std'].mean(),
            'min_polarization': min_polar,
            'max_polarization': max_polar,
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

# REMOVED: create_2d_phase_diagram - unused visualization

# REMOVED: create_performance_landscape_plot - unused visualization

# REMOVED: create_continuous_landscape_plot - unused visualization

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
    ax1.set_ylabel('$\\langle a^* \\rangle$', fontsize=FONT_SIZE-1)
    ax1.grid(True, alpha=0.3, linewidth=0.3)
    ax1.tick_params(labelsize=FONT_SIZE-2)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              fontsize=FONT_SIZE-2, title='Algorithm')
    
    # Customize polarization plot
    ax2.set_xlabel('Stubbornness, $w_i$', fontsize=FONT_SIZE-1)
    ax2.set_ylabel('$\\langle P^* \\rangle$', fontsize=FONT_SIZE-1)
    ax2.grid(True, alpha=0.3, linewidth=0.3)
    ax2.tick_params(labelsize=FONT_SIZE-2)
    
    plt.tight_layout()
    return fig

def create_continuous_single_panel_plot(df, for_combined=False, show_uncertainty_bands=False):
    """Create single-panel plot matching stubbornness_trajectories.pdf data structure

    Args:
        df: DataFrame with regime data
        for_combined: If True, adjust figure size for combined panel figure
        show_uncertainty_bands: If True, show shaded bands for WTF indicating range across backfirer fractions
    """
    setup_style()

    # Create figure with single panel and dual y-axes
    # Use matching height when combining with heatmap
    figsize = (8*cm, 6*cm) if for_combined else (8*cm, 6*cm)
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)

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

    # First, add shaded band for WTF if requested (do this first so it appears behind lines)
    if show_uncertainty_bands:
        wtf_data = df_main[df_main['algorithm'] == 'WTF'].sort_values('stubbornness_numeric')
        if len(wtf_data) > 0 and 'min_cooperation' in wtf_data.columns:
            color = algorithm_colors.get('WTF', '#666666')
            # Cooperation band
            ax1.fill_between(wtf_data['stubbornness_numeric'],
                             wtf_data['min_cooperation'],
                             wtf_data['max_cooperation'],
                             color=color, alpha=0.15, linewidth=0)
            # Polarization band
            ax2.fill_between(wtf_data['stubbornness_numeric'],
                             wtf_data['min_polarization'],
                             wtf_data['max_polarization'],
                             color=color, alpha=0.1, linewidth=0)

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
    # Match heatmap font size (BASE_FONT_SIZE - 1 = 7) for combined figure consistency
    xlabel_fontsize = 7 if for_combined else FONT_SIZE-1
    ylabel_fontsize = 7 if for_combined else FONT_SIZE-1
    ax1.set_xlabel('Stubbornness, $w_i$', fontsize=xlabel_fontsize, labelpad=3)
    ax1.set_ylabel('$\\langle a^* \\rangle$', fontsize=ylabel_fontsize, labelpad=3, color='black')
    ax1.grid(True, alpha=0.3, linewidth=0.3)
    ax1.tick_params(labelsize=FONT_SIZE-2, length=2, width=0.5)

    # Customize right axis (polarization)
    ax2.set_ylabel('$\\langle P^* \\rangle$', fontsize=ylabel_fontsize, labelpad=3, color='black')
    ax2.tick_params(labelsize=FONT_SIZE-2, length=2, width=0.5)
    
    # Add regime boundary lines (but don't label them as regimes)
    ax1.axvline(0.4, color='black', linestyle='--', alpha=0.7, linewidth=0.8)
    ax1.axvline(0.7, color='black', linestyle='--', alpha=0.7, linewidth=0.8)
    
    # Add regime labels at top (bigger font)
    ax1.text(0.2, ax1.get_ylim()[1]*1.12, 'Low', ha='center',
            fontsize=FONT_SIZE-1, color='black')
    ax1.text(0.55, ax1.get_ylim()[1]*1.12, 'Medium', ha='center',
            fontsize=FONT_SIZE-1, color='black')
    ax1.text(0.85, ax1.get_ylim()[1]*1.12, 'High', ha='center',
            fontsize=FONT_SIZE-1, color='black')

    # Legend at bottom with reduced spacing
    fig.legend(bbox_to_anchor=(0.5, -0.02), loc='upper center', ncol=3,
              fontsize=FONT_SIZE-2, frameon=True, fancybox=False,
              edgecolor='black', facecolor='white')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Reduced from 0.25 to minimize whitespace
    
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

    ax1.set_xlabel('Stubbornness, $w_i$', fontsize=FONT_SIZE-1)
    ax1.set_ylabel('$\\langle a^* \\rangle$', fontsize=FONT_SIZE-1)
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

    ax2.set_xlabel('Stubbornness, $w_i$', fontsize=FONT_SIZE-1)
    ax2.set_ylabel('$\\langle P^* \\rangle$', fontsize=FONT_SIZE-1)
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

def create_backfirer_parameter_space(df):
    """Create parameter space plot showing backfirer fraction as continuous variable"""
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

    # Panel 1: Backfirer Fraction vs Cooperation
    for alg in main_algorithms:
        alg_data = df_main[df_main['algorithm'] == alg]
        if len(alg_data) > 0:
            color = algorithm_colors.get(alg, '#666666')
            ax1.scatter(alg_data['backfirer_fraction'], alg_data['mean_cooperation'],
                       color=color, alpha=0.8, s=25, label=alg)
            # Fit trend line
            if len(alg_data) >= 2:
                bf_sorted = alg_data.sort_values('backfirer_fraction')
                ax1.plot(bf_sorted['backfirer_fraction'],
                        bf_sorted['mean_cooperation'],
                        color=color, linewidth=1, alpha=0.7)

    ax1.set_xlabel('Backfirer Fraction, $\\rho$', fontsize=FONT_SIZE-1)
    ax1.set_ylabel('$\\langle a^* \\rangle$', fontsize=FONT_SIZE-1)
    ax1.grid(True, alpha=0.3, linewidth=0.3)
    ax1.tick_params(labelsize=FONT_SIZE-2)
    ax1.set_title('A', fontsize=FONT_SIZE, fontweight='bold')

    # Panel 2: Backfirer Fraction vs Polarization
    for alg in main_algorithms:
        alg_data = df_main[df_main['algorithm'] == alg]
        if len(alg_data) > 0:
            color = algorithm_colors.get(alg, '#666666')
            ax2.scatter(alg_data['backfirer_fraction'], alg_data['mean_polarization'],
                       color=color, alpha=0.8, s=25)
            if len(alg_data) >= 2:
                bf_sorted = alg_data.sort_values('backfirer_fraction')
                ax2.plot(bf_sorted['backfirer_fraction'],
                        bf_sorted['mean_polarization'],
                        color=color, linewidth=1, alpha=0.7)

    ax2.set_xlabel('Backfirer Fraction, $\\rho$', fontsize=FONT_SIZE-1)
    ax2.set_ylabel('$\\langle P^* \\rangle$', fontsize=FONT_SIZE-1)
    ax2.grid(True, alpha=0.3, linewidth=0.3)
    ax2.tick_params(labelsize=FONT_SIZE-2)
    ax2.set_title('B', fontsize=FONT_SIZE, fontweight='bold')

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
            ax_coop.text(0.5, 1.15, '$\\langle a^* \\rangle$', transform=ax_coop.transAxes,
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
            ax_polar.text(0.5, 1.15, '$\\langle P^* \\rangle$', transform=ax_polar.transAxes,
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

# ============================================================================
# REGIME-BASED ANALYSIS FUNCTIONS (from regime_performance_panels.py)
# ============================================================================

# Marker styles for topologies (matching existing style)
TOPOLOGY_MARKERS = {'DPAH': 'x', 'cl': '+', 'Twitter': '^', 'FB': 'o'}

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

def create_regime_three_panel_plot(algorithm_data):
    """Create three-panel plot showing regime performance"""
    setup_style()

    # Create figure with two panels (combining cooperation and polarization)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12*cm, 6*cm))

    regimes = ['low', 'medium', 'high']
    regime_labels = ['Low\n(< 0.4)', 'Medium\n(0.4-0.7)', 'High\n(> 0.7)']
    x_pos = np.arange(len(regimes))

    # Get all algorithms (use medium regime as reference)
    algorithms = algorithm_data['medium']['algorithms']

    # Create color mapping for algorithms
    algorithm_colors = {}
    for alg in algorithms:
        alg_lower = alg.lower()
        if alg_lower == 'opposite':
            algorithm_colors[alg] = FRIENDLY_COLORS['local (opposite)']
        elif alg_lower == 'similar':
            algorithm_colors[alg] = FRIENDLY_COLORS['local (similar)']
        elif alg_lower == 'wtf':
            algorithm_colors[alg] = FRIENDLY_COLORS['empirical wtf']
        elif alg_lower == 'node2vec':
            algorithm_colors[alg] = FRIENDLY_COLORS['empirical node2vec']
        elif alg_lower == 'static':
            algorithm_colors[alg] = FRIENDLY_COLORS['static']
        elif alg_lower == 'random':
            algorithm_colors[alg] = FRIENDLY_COLORS['random']
        else:
            algorithm_colors[alg] = '#666666'

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
    ax1_twin.set_ylabel('$\\langle P^* \\rangle$', fontsize=FONT_SIZE-1, labelpad=2, color='black')
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

def create_regime_single_panel_plot(algorithm_data):
    """Create single-panel plot showing cooperation and polarization across regimes (Panel A only)"""
    setup_style()

    # Create figure with single panel
    fig, ax1 = plt.subplots(1, 1, figsize=(8*cm, 6*cm))

    regimes = ['low', 'medium', 'high']
    regime_labels = ['Low\n(< 0.4)', 'Medium\n(0.4-0.7)', 'High\n(> 0.7)']
    x_pos = np.arange(len(regimes))

    # Get all algorithms (use medium regime as reference)
    algorithms = algorithm_data['medium']['algorithms']

    # Create color mapping for algorithms
    algorithm_colors = {}
    for alg in algorithms:
        alg_lower = alg.lower()
        if alg_lower == 'opposite':
            algorithm_colors[alg] = FRIENDLY_COLORS['local (opposite)']
        elif alg_lower == 'similar':
            algorithm_colors[alg] = FRIENDLY_COLORS['local (similar)']
        elif alg_lower == 'wtf':
            algorithm_colors[alg] = FRIENDLY_COLORS['empirical wtf']
        elif alg_lower == 'node2vec':
            algorithm_colors[alg] = FRIENDLY_COLORS['empirical node2vec']
        elif alg_lower == 'static':
            algorithm_colors[alg] = FRIENDLY_COLORS['static']
        elif alg_lower == 'random':
            algorithm_colors[alg] = FRIENDLY_COLORS['random']
        else:
            algorithm_colors[alg] = '#666666'

    # Create twin axis for polarization
    ax1_twin = ax1.twinx()

    # Plot each algorithm across regimes
    for i, alg in enumerate(algorithms):
        # Extract values for this algorithm across regimes
        coop_vals = [algorithm_data[regime]['cooperation'][i] for regime in regimes]
        polar_vals = [algorithm_data[regime]['polarization'][i] for regime in regimes]

        color = algorithm_colors.get(alg, '#666666')

        # Mean Cooperation (solid lines) and Polarization (dotted lines)
        ax1.plot(x_pos, coop_vals, 'o-', color=color, linewidth=1,
                markersize=4, alpha=0.8, label=alg)
        ax1_twin.plot(x_pos, polar_vals, 'o:', color=color, linewidth=1,
                markersize=4, alpha=0.6)

    # Customize Panel - Cooperation & Polarization Combined with dual y-axes
    ax1.set_xlabel('Stubbornness Regime', fontsize=FONT_SIZE-1, labelpad=2)
    ax1.set_ylabel('$a$', fontsize=FONT_SIZE-1, labelpad=2, color='black')
    ax1_twin.set_ylabel('$\\langle P^* \\rangle$', fontsize=FONT_SIZE-1, labelpad=2, color='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(regime_labels, fontsize=FONT_SIZE-2)
    ax1.grid(True, alpha=0.3, linewidth=0.3)
    ax1.tick_params(labelsize=FONT_SIZE-2)
    ax1_twin.tick_params(labelsize=FONT_SIZE-2)

    # Add legend at the bottom
    fig.legend(bbox_to_anchor=(0.5, 0.02), loc='upper center', ncol=3,
              fontsize=FONT_SIZE-2, frameon=True, fancybox=False,
              edgecolor='black', facecolor='white')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for legend

    return fig

# ============================================================================
# BACKFIRER FRACTION ANALYSIS FUNCTIONS (from regime_performance_panels.py)
# ============================================================================

def load_continuous_backfirer_data(min_stubbornness=None, max_stubbornness=None):
    """Load continuous parameter data from heatmap files and calculate averages per backfirer fraction value

    Args:
        min_stubbornness: Minimum stubbornness to include (default None = include all)
        max_stubbornness: Maximum stubbornness to include (default None = include all)
    """
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
    if 'polarisingNode_f' not in df.columns:
        print(f"Error: polarisingNode_f column not found. Available columns: {list(df.columns)}")
        return None

    # Filter for stubbornness range if specified
    if 'stubbornness' in df.columns:
        if min_stubbornness is not None:
            df = df[df['stubbornness'] >= min_stubbornness]
            print(f"Filtered to stubbornness >= {min_stubbornness}: {len(df)} records remaining")
        if max_stubbornness is not None:
            df = df[df['stubbornness'] <= max_stubbornness]
            print(f"Filtered to stubbornness <= {max_stubbornness}: {len(df)} records remaining")
    else:
        print("Warning: stubbornness column not found, cannot filter by stubbornness")

    # Create scenario column from rewiring and mode
    df['rewiring'] = df['rewiring'].fillna('none')
    df['mode'] = df['mode'].fillna('none')
    df['scenario'] = df['rewiring'] + ' ' + df['mode']

    # Create cleaner algorithm names
    algorithm_mapping = {
        'diff biased': 'Opposite',
        'same biased': 'Similar',
        'none wtf': 'WTF',
        'none node2vec': 'Node2Vec',
        'none none': 'Static',
        'none random': 'Random',
        'diff bridge': 'Bridge (Opposite)',
        'same bridge': 'Bridge (Similar)'
    }
    df['algorithm'] = df['scenario'].map(algorithm_mapping).fillna(df['scenario'])

    # Group by backfirer fraction and algorithm, calculate averages AND ranges
    aggregated_data = []

    for (bf_val, algorithm), group in df.groupby(['polarisingNode_f', 'algorithm']):
        # Calculate cooperative states (state > 0)
        cooperative_mask = group['state'] > 0
        n_cooperative = cooperative_mask.sum()
        n_total = len(group)

        # For shaded bands: calculate min/max cooperation across different stubbornness values
        # Group by stubbornness first to get one value per stubbornness level
        if 'stubbornness' in group.columns:
            stub_group_coop = group.groupby('stubbornness')['state'].mean()
            stub_group_polar = group.groupby('stubbornness')['state_std'].mean()
            min_coop = stub_group_coop.min()
            max_coop = stub_group_coop.max()
            min_polar = stub_group_polar.min()
            max_polar = stub_group_polar.max()
        else:
            min_coop = group['state'].min()
            max_coop = group['state'].max()
            min_polar = group['state_std'].min()
            max_polar = group['state_std'].max()

        metrics = {
            'backfirer_fraction': bf_val,
            'algorithm': algorithm,
            'mean_cooperation': group['state'].mean(),
            'min_cooperation': min_coop,
            'max_cooperation': max_coop,
            'mean_polarization': group['state_std'].mean(),
            'min_polarization': min_polar,
            'max_polarization': max_polar,
            'cooperative_volume_percent': (n_cooperative / n_total) * 100 if n_total > 0 else 0.0,
            'n_runs': n_total
        }
        aggregated_data.append(metrics)

    result_df = pd.DataFrame(aggregated_data)
    print(f"Aggregated to {len(result_df)} (backfirer_fraction, algorithm) combinations")
    print(f"Algorithms: {sorted(result_df['algorithm'].unique())}")
    print(f"Backfirer fraction values: {sorted(result_df['backfirer_fraction'].unique())}")

    return result_df

def create_continuous_backfirer_single_panel_plot(df):
    """Create single-panel plot with backfirer fraction as continuous x-axis

    Includes shaded bands showing range across stubbornness values
    Mirrors the structure of create_continuous_single_panel_plot but for backfirer fraction
    """
    setup_style()

    # Create figure with single panel and dual y-axes
    fig, ax1 = plt.subplots(1, 1, figsize=(8.7*cm, 6*cm))

    # Filter main algorithms
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

    # Create twin axis for polarization
    ax1_twin = ax1.twinx()

    # First, add shaded band for WTF (do this first so it appears behind lines)
    wtf_data = df_main[df_main['algorithm'] == 'WTF'].sort_values('backfirer_fraction')
    if len(wtf_data) > 0 and 'min_cooperation' in wtf_data.columns:
        color = algorithm_colors.get('WTF', '#666666')
        # Cooperation band
        ax1.fill_between(wtf_data['backfirer_fraction'],
                         wtf_data['min_cooperation'],
                         wtf_data['max_cooperation'],
                         color=color, alpha=0.15, linewidth=0)
        # Polarization band
        ax1_twin.fill_between(wtf_data['backfirer_fraction'],
                             wtf_data['min_polarization'],
                             wtf_data['max_polarization'],
                             color=color, alpha=0.1, linewidth=0)

    # Plot each algorithm trajectory across backfirer fraction values
    for alg in main_algorithms:
        alg_data = df_main[df_main['algorithm'] == alg].sort_values('backfirer_fraction')
        if len(alg_data) > 0:
            color = algorithm_colors.get(alg, '#666666')

            # Mean Cooperation (solid lines) and Polarization (dotted lines)
            ax1.plot(alg_data['backfirer_fraction'], alg_data['mean_cooperation'],
                    'o-', color=color, linewidth=1, markersize=2.5, alpha=0.8, label=alg)
            ax1_twin.plot(alg_data['backfirer_fraction'], alg_data['mean_polarization'],
                         'o:', color=color, linewidth=1, markersize=2.5, alpha=0.6)

    # Customize Panel - Cooperation & Polarization with dual y-axes
    ax1.set_xlabel('Backfirer Fraction ($\\rho$)', fontsize=FONT_SIZE-1, labelpad=2)
    ax1.set_ylabel('$a$', fontsize=FONT_SIZE-1, labelpad=2, color='black')
    ax1_twin.set_ylabel('$\\langle P^* \\rangle$', fontsize=FONT_SIZE-1, labelpad=2, color='black')
    ax1.grid(True, alpha=0.3, linewidth=0.3)
    ax1.tick_params(labelsize=FONT_SIZE-2)
    ax1_twin.tick_params(labelsize=FONT_SIZE-2)

    # Add legend at the bottom
    fig.legend(bbox_to_anchor=(0.5, 0.02), loc='upper center', ncol=3,
              fontsize=FONT_SIZE-2, frameon=True, fancybox=False,
              edgecolor='black', facecolor='white')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend

    return fig

def prepare_backfirer_regime_data(df_continuous=None):
    """Prepare algorithm data organized by backfirer fraction levels using continuous data

    Args:
        df_continuous: DataFrame from load_continuous_backfirer_data() with continuous backfirer fraction data
                      If None, will attempt to load from comprehensive comparison files (legacy behavior)
    """
    # If continuous data provided, use it directly
    if df_continuous is not None:
        # Filter to main algorithms
        main_algorithms = ['Opposite', 'Similar', 'WTF', 'Node2Vec', 'Static', 'Random']
        df = df_continuous[df_continuous['algorithm'].isin(main_algorithms)].copy()

        # Determine thresholds using tertiles
        bf_values = df['backfirer_fraction'].unique()
        bf_values.sort()

        if len(bf_values) > 0:
            min_val, max_val = bf_values.min(), bf_values.max()
            low_threshold = min_val + (max_val - min_val) / 3
            high_threshold = min_val + 2 * (max_val - min_val) / 3
        else:
            low_threshold = 0.33
            high_threshold = 0.67

        print(f"Using backfirer thresholds: Low ≤ {low_threshold:.3f}, Medium {low_threshold:.3f}-{high_threshold:.3f}, High > {high_threshold:.3f}")

        # Initialize data structure
        backfirer_regime_data = {}
        for regime_name in ['low_backfirer', 'medium_backfirer', 'high_backfirer']:
            backfirer_regime_data[regime_name] = {
                'algorithms': main_algorithms.copy(),
                'cooperation': [],
                'polarization': [],
                'cooperative_volume': []
            }

        # Classify data points into regimes and aggregate by algorithm
        for alg in main_algorithms:
            alg_data = df[df['algorithm'] == alg]

            for regime_key in ['low_backfirer', 'medium_backfirer', 'high_backfirer']:
                # Filter by regime
                if regime_key == 'low_backfirer':
                    regime_data = alg_data[alg_data['backfirer_fraction'] <= low_threshold]
                elif regime_key == 'medium_backfirer':
                    regime_data = alg_data[(alg_data['backfirer_fraction'] > low_threshold) &
                                          (alg_data['backfirer_fraction'] <= high_threshold)]
                else:  # high_backfirer
                    regime_data = alg_data[alg_data['backfirer_fraction'] > high_threshold]

                # Calculate averages for this regime
                if len(regime_data) > 0:
                    avg_coop = regime_data['mean_cooperation'].mean()
                    avg_polar = regime_data['mean_polarization'].mean()
                    avg_volume = regime_data['cooperative_volume_percent'].mean()
                else:
                    avg_coop = np.nan
                    avg_polar = np.nan
                    avg_volume = np.nan

                backfirer_regime_data[regime_key]['cooperation'].append(avg_coop)
                backfirer_regime_data[regime_key]['polarization'].append(avg_polar)
                backfirer_regime_data[regime_key]['cooperative_volume'].append(avg_volume)

        return backfirer_regime_data, low_threshold, high_threshold

    # Legacy behavior: Load from comprehensive comparison files
    base_dir = "../../Output/Stats/stubborness_backfirer"

    # Load all regime data first - use comprehensive files for consistency
    all_regime_data = {}
    for regime_key in ['low', 'medium', 'high']:
        # Try multiple date patterns
        for date_pattern in ['20251023', '20250925', '20250924']:
            file_path = f"{base_dir}/comprehensive_algorithm_comparison_{regime_key}_{date_pattern}.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                all_regime_data[regime_key] = df
                break

    # Get all algorithms and their backfirer fractions across regimes
    if 'medium' not in all_regime_data:
        print("Warning: Could not find comprehensive algorithm comparison files")
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

    print(f"Using backfirer thresholds: Low ≤ {low_threshold:.3f}, Medium {low_threshold:.3f}-{high_threshold:.3f}, High > {high_threshold:.3f}")

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

    # Get all algorithms
    algorithms = backfirer_data['low_backfirer']['algorithms']

    # Create color mapping for algorithms
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
            algorithm_colors[alg] = '#666666'

    # Plot each algorithm across backfirer regimes
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

    # Customize Panel A
    ax1.set_xlabel('Backfirer Regime', fontsize=FONT_SIZE-1, labelpad=2)
    ax1.set_ylabel('$a$', fontsize=FONT_SIZE-1, labelpad=2, color='black')
    ax1_twin.set_ylabel('$\\langle P^* \\rangle$', fontsize=FONT_SIZE-1, labelpad=2, color='black')
    ax1.set_title('A', fontsize=FONT_SIZE, pad=5, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(regime_labels, fontsize=FONT_SIZE-2)
    ax1.grid(True, alpha=0.3, linewidth=0.3)
    ax1.tick_params(labelsize=FONT_SIZE-2)
    ax1_twin.tick_params(labelsize=FONT_SIZE-2)

    # Customize Panel B
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

# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

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

    # Generate stubbornness trajectory plot
    print("Creating stubbornness trajectory plot...")
    fig1 = create_stubbornness_trajectory_plot(df)
    output_path1 = f"{output_dir}/stubbornness_trajectories_{today}.pdf"
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"Trajectory plot saved: {output_path1}")
    plt.show()
    plt.close()

    # Generate continuous single panel plot
    print("Creating continuous single panel plot...")
    fig2 = create_continuous_single_panel_plot(df)
    output_path2 = f"{output_dir}/continuous_single_panel_{today}.pdf"
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Single panel plot saved: {output_path2}")
    plt.show()
    plt.close()

    # Generate stubbornness parameter space
    print("Creating stubbornness parameter space...")
    fig3 = create_stubbornness_parameter_space(df)
    output_path3 = f"{output_dir}/stubbornness_parameter_space_{today}.pdf"
    fig3.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"Parameter space plot saved: {output_path3}")
    plt.show()
    plt.close()

    print("Consolidated continuous visualization generation complete!")

def main_regime():
    """Generate regime-based performance visualization (discrete low/medium/high regimes)"""
    print("Creating regime-based performance visualization...")

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
    fig1 = create_regime_three_panel_plot(algorithm_data)
    output_path1 = f"{output_dir}/regime_performance_comprehensive_{today}.pdf"
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"Comprehensive plot saved: {output_path1}")
    plt.show()
    plt.close()

    # Generate single-panel plot (Panel A only) for main text
    print("Creating single-panel plot (Panel A only)...")
    fig2 = create_regime_single_panel_plot(algorithm_data)
    output_path2 = f"{output_dir}/regime_performance_single_panel_{today}.pdf"
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Single-panel plot saved: {output_path2}")
    plt.show()
    plt.close()

    print("Regime visualization complete!")

    # Print some key insights
    print("\n=== KEY INSIGHTS ==")
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

def main_backfirer():
    """Generate backfirer fraction analysis visualizations"""
    print("Creating backfirer fraction analysis visualizations...")

    # Load continuous backfirer data (averaging over all stubbornness values)
    df = load_continuous_backfirer_data()

    if df is None:
        print("No data found. Please check heatmap sweep files.")
        return

    print(f"Loaded data for algorithms: {sorted(df['algorithm'].unique())}")
    print(f"Backfirer fraction range: {df['backfirer_fraction'].min():.3f} - {df['backfirer_fraction'].max():.3f}")

    # Create output directory
    output_dir = "../../Figs/Regime_Analysis"
    os.makedirs(output_dir, exist_ok=True)
    today = date.today().strftime("%Y%m%d")

    # Generate continuous single-panel plot for backfirer fraction
    print("Creating continuous backfirer single-panel plot...")
    fig1 = create_continuous_backfirer_single_panel_plot(df)
    output_path1 = f"{output_dir}/backfirer_performance_single_panel_continuous_{today}.pdf"
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"Continuous backfirer plot saved: {output_path1}")
    plt.show()
    plt.close()

    # Generate backfirer regime plot
    print("Creating backfirer regime analysis...")
    backfirer_data, low_thresh, high_thresh = prepare_backfirer_regime_data(df)
    fig2 = create_backfirer_regime_plot(backfirer_data, low_thresh, high_thresh)
    output_path2 = f"{output_dir}/backfirer_regimes_{today}.pdf"
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Backfirer regime plot saved: {output_path2}")
    plt.show()
    plt.close()

    # Generate backfirer parameter space plot
    print("Creating backfirer parameter space...")
    fig3 = create_backfirer_parameter_space(df)
    output_path3 = f"{output_dir}/backfirer_parameter_space_{today}.pdf"
    fig3.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"Backfirer parameter space plot saved: {output_path3}")
    plt.show()
    plt.close()

    print("Backfirer visualization complete!")

    # Print key insights
    print("\n=== KEY INSIGHTS (CONTINUOUS BACKFIRER FRACTION) ===")
    for bf_val in [0.0, 0.2, 0.4]:
        closest_data = df[df['backfirer_fraction'] == df.iloc[(df['backfirer_fraction'] - bf_val).abs().argsort()[:1]]['backfirer_fraction'].iloc[0]]
        if len(closest_data) > 0:
            best_idx = closest_data['mean_cooperation'].idxmax()
            worst_idx = closest_data['mean_cooperation'].idxmin()

            print(f"Backfirer fraction ≈ {bf_val}:")
            print(f"  Best: {df.loc[best_idx, 'algorithm']} ({df.loc[best_idx, 'mean_cooperation']:.3f})")
            print(f"  Worst: {df.loc[worst_idx, 'algorithm']} ({df.loc[worst_idx, 'mean_cooperation']:.3f})")

            # Find WTF performance
            wtf_data = closest_data[closest_data['algorithm'] == 'WTF']
            if len(wtf_data) > 0:
                wtf_coop = wtf_data['mean_cooperation'].iloc[0]
                print(f"  WTF: {wtf_coop:.3f}")

if __name__ == "__main__":
    import sys

    # Parse command-line arguments for different analysis modes
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "--regime":
            main_regime()
        elif mode == "--backfirer":
            main_backfirer()
        elif mode == "--continuous":
            main()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python continuous_regime_analysis.py [--continuous|--regime|--backfirer]")
            print("  --continuous (default): Continuous stubbornness parameter analysis")
            print("  --regime: Regime-based analysis (low/medium/high)")
            print("  --backfirer: Backfirer fraction analysis")
    else:
        # Default: run continuous analysis (for backward compatibility)
        main()