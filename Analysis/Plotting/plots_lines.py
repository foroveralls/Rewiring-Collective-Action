import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
from matplotlib.gridspec import GridSpec

# Configuration
cm = 1/2.54
FONT_SIZE = 7
LINE_WIDTH = 0.5
COLORS = {
    'none_none': '#EE7733', 'random_none': '#0077BB', 'biased_same': '#33BBEE',
    'biased_diff': '#009988', 'bridge_same': '#CC3311', 'bridge_diff': '#EE3377',
    'wtf_none': '#BBBBBB', 'node2vec_none': '#44BB99'
}
NETWORKS = {'cl': 'CSF', 'DPAH': 'DPAH', 'Twitter': 'Twitter', 'FB': 'FB'}
LABELS = {
    'none_none': 'static', 'random_none': 'random', 'biased_same': 'local (similar)',
    'biased_diff': 'local (opposite)', 'bridge_same': 'bridge (similar)', 
    'bridge_diff': 'bridge (opposite)', 'wtf_none': 'wtf', 'node2vec_none': 'node2vec'
}

def set_plot_style():
    sns.set_style("white")
    plt.rcParams.update({
        'font.size': FONT_SIZE, 'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE, 'ytick.labelsize': FONT_SIZE, 'axes.linewidth': 0.8,
        'lines.linewidth': 1.5, 'figure.dpi': 300, 'savefig.dpi': 300,
        'figure.figsize': (17.8*cm, 8.9*cm), 'grid.alpha': 0.4, 'grid.linestyle': '--',
        'mathtext.default': 'regular', 'axes.formatter.use_mathtext': True, 'axes.axisbelow': True
    })

def process_data(data, t_max, scale_type='linear'):
    req_cols = ['t', 'avg_state', 'std_states', 'scenario', 'rewiring', 'type']
    for col in req_cols:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found")
    
    data = data[data['t'] <= t_max].copy()
    if scale_type == 'log':
        data = data[data['t'] > 0].copy()
    
    data['rewiring'] = data['rewiring'].fillna('none')
    data['scenario'] = data['scenario'].fillna('none')
    data['scenario_grouped'] = data['scenario'].str.cat(data['rewiring'], sep='_')
    
    return pd.melt(data.drop(columns=['scenario', 'rewiring']).rename(columns={'std_states': 'polarization'}),
                   id_vars=['t', 'type', 'scenario_grouped'], 
                   value_vars=['avg_state', 'polarization'], 
                   var_name='measure', value_name='value')

def configure_axis(ax, t_max, scale_type='linear', show_ylabel=True):
    ax.set_ylim(-0.6, 1.1)
    
    if scale_type == 'log':
        ax.set_xscale('log')
        ax.set_xlim(1, t_max)
        xlabel = "Time, t (log scale)"
    elif scale_type == 'symlog':
        ax.set_xscale('symlog', linthresh=30000)
        ax.set_xlim(0, t_max)
        xlabel = "Time, t (symlog)"
    else:
        ax.set_xlim(0, t_max)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 1))
        ax.xaxis.set_major_formatter(formatter)
        
        # Create nice round tick intervals
        step = 10000 if t_max <= 60000 else 20000 if t_max <= 120000 else 50000
        ticks = np.arange(0, t_max + step, step)
        ax.set_xticks(ticks[ticks <= t_max])
        xlabel = "Time, t"
    
    ax.grid(True, alpha=0.4, linestyle='--', zorder=5)
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_zorder(100)
    
    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    
    if scale_type == 'linear':
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    for axis in ['x', 'y']:
        ax.tick_params(axis=axis, which='major', direction='out', length=2, width=0.8, 
                      colors='black', zorder=100)
        ax.tick_params(axis=axis, which='minor', direction='out', length=1, width=0.6, 
                      colors='black', zorder=100)
    
    if show_ylabel:
        ax.set_ylabel("Cooperativity, ⟨a⟩")
    ax.set_xlabel(xlabel)
    
    return xlabel

def add_legends(fig, data, legend_type='full'):
    if legend_type == 'full':
        # Top legend
        legend_ax = fig.add_axes([0.15, 0.90, 0.7, 0.04])
        legend_ax.axis('off')
        line_elements = [
            Line2D([], [], color='black', linestyle='-', label='cooperativity'),
            Line2D([], [], color='black', linestyle='--', dashes=(4, 2), label='polarization')
        ]
        legend_ax.legend(handles=line_elements, ncol=2, loc='center', frameon=True, bbox_to_anchor=(0.5, 0.5))
        
        # Bottom legend
        bottom_ax = fig.add_axes([0.15, 0.03, 0.7, 0.05])
        bottom_ax.axis('off')
        color_elements = [Line2D([], [], color=COLORS[k], label=v) for k, v in LABELS.items() 
                         if any(k in s for s in data['scenario_grouped'])]
        bottom_ax.legend(handles=color_elements, ncol=4, loc='center', frameon=True, bbox_to_anchor=(0.5, 0.5))
    
    else:  # compact legend
        legend_ax = fig.add_axes([0.1, 0.95, 0.8, 0.05])
        legend_ax.axis('off')
        all_elements = [
            Line2D([], [], color='black', linestyle='-', label='cooperativity'),
            Line2D([], [], color='black', linestyle='--', dashes=(4, 2), label='polarization'),
            Line2D([], [], color='black', marker='>', markersize=2, label='directed'),
            Line2D([], [], color='black', label='undirected')
        ]
        legend_ax.legend(handles=all_elements, ncol=5, loc='center', frameon=True, 
                        bbox_to_anchor=(0.5, 0.5), fontsize=FONT_SIZE-2)

def plot_network_dynamics(data, t_max=55000, scale_type='linear', output_file=None):
    scenario_colors = {s: COLORS.get('_'.join(s.split('_')[:2]).lower(), '#FE6900') 
                      for s in data['scenario_grouped'].unique()}
    
    avg_data = data[data['measure'] == 'avg_state']
    pol_data = data[data['measure'] == 'polarization']
    
    g = sns.relplot(data=avg_data, x='t', y='value', hue='scenario_grouped',
                   col='type', linewidth=LINE_WIDTH, kind='line', col_wrap=2, 
                   height=4, aspect=1, palette=scenario_colors, legend=False)
    
    g.fig.set_size_inches(11.8*cm, 11*cm)
    
    # Add polarization lines
    for i, ax in enumerate(g.axes.flat):
        network = list(avg_data['type'].unique())[i]
        for scenario in pol_data['scenario_grouped'].unique():
            pol_subset = pol_data[(pol_data['scenario_grouped'] == scenario) & 
                                (pol_data['type'] == network)]
            if not pol_subset.empty:
                ax.plot(pol_subset['t'], pol_subset['value'], linestyle='--', 
                       dashes=(4, 2), color=scenario_colors[scenario], linewidth=LINE_WIDTH)
    
    # Configure axes
    xlabel = configure_axis(g.axes[0], t_max, scale_type, True)
    for i, ax in enumerate(g.axes.flat):
        network = list(avg_data['type'].unique())[i]
        ax.set_title(NETWORKS.get(network, network))
        configure_axis(ax, t_max, scale_type, i % 2 == 0)  # ylabel only on left column
        
        if i < 2:  # top row
            ax.set_xlabel("")
            if scale_type == 'linear':
                ax.xaxis.offsetText.set_visible(False)
    
    g.set_axis_labels(xlabel, "Cooperativity, ⟨a⟩")
    g.fig.subplots_adjust(top=0.84, bottom=0.20, hspace=0.45, wspace=0.28, left=0.1, right=0.95)
    
    add_legends(g.fig, data, 'full')
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    return g

def plot_single_topology_dynamics(data, t_max=55000, scale_type='linear', output_file=None):
    categories = {'none_none': 'static', 'random_none': 'random', 'biased_same': 'local',    
                 'biased_diff': 'local', 'bridge_same': 'bridge', 'bridge_diff': 'bridge',  
                 'wtf_none': 'wtf', 'node2vec_none': 'node2vec'}
    
    dpah_data = data[data['type'] == 'DPAH']
    cl_data = data[data['type'] == 'cl']
    
    if scale_type == 'log':
        dpah_data = dpah_data[dpah_data['t'] > 0]
        cl_data = cl_data[cl_data['t'] > 0]
    
    # Group scenarios by category
    plots = {}
    for cat, label in {'static': 'A', 'random': 'B', 'local': 'C', 
                      'bridge': 'D', 'wtf': 'E', 'node2vec': 'F'}.items():
        scenarios = [s for s in data['scenario_grouped'].unique() 
                    if categories.get('_'.join(s.split('_')[:2]).lower()) == cat]
        if scenarios:
            plots[label] = scenarios
    
    n = len(plots)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig = plt.figure(figsize=(17.8*cm, 12*cm))
    plt.subplots_adjust(top=0.90, bottom=0.16, hspace=0.27, wspace=0.22, left=0.1, right=0.95)
    
    add_legends(fig, data, 'compact')
    
    # Get static reference data
    static_dpah = dpah_data[dpah_data['scenario_grouped'].str.startswith('none_none')]
    static_cl = cl_data[cl_data['scenario_grouped'].str.startswith('none_none')]
    
    gs = GridSpec(rows, cols, figure=fig, top=0.90, bottom=0.16, hspace=0.27, wspace=0.22, left=0.1, right=0.95)
    
    for i, (key, scenarios) in enumerate(plots.items()):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        is_static = key == 'A'
        
        # Plot data
        for measure in ['avg_state', 'polarization']:
            linestyle = '-' if measure == 'avg_state' else '--'
            dashes = None if measure == 'avg_state' else (4, 2)
            
            # Plot static reference for both networks
            for data_subset, is_directed in [(dpah_data, True), (cl_data, False)]:
                static_subset = static_dpah if is_directed else static_cl
                static_measure = static_subset[static_subset['measure'] == measure]
                
                if not static_measure.empty:
                    static_avg = static_measure.groupby('t')['value'].mean()
                    props = {'color': COLORS['none_none'], 'linestyle': linestyle, 
                           'linewidth': LINE_WIDTH, 'alpha': 0.7 if not is_static else 1.0}
                    if dashes:
                        props['dashes'] = dashes
                    if is_directed:
                        props.update({'marker': '>', 'markersize': 2, 'markevery': 0.1})
                    ax.plot(static_avg.index, static_avg.values, **props)
            
            # Plot scenario data
            if not is_static:
                for scenario in scenarios:
                    color_key = '_'.join(scenario.split('_')[:2]).lower()
                    for data_subset, is_directed in [(dpah_data, True), (cl_data, False)]:
                        subset = data_subset[(data_subset['scenario_grouped'] == scenario) & 
                                           (data_subset['measure'] == measure)]
                        if not subset.empty:
                            avg = subset.groupby('t')['value'].mean()
                            props = {'color': COLORS.get(color_key, '#FE6900'), 'linestyle': linestyle, 
                                   'linewidth': LINE_WIDTH}
                            if dashes:
                                props['dashes'] = dashes
                            if is_directed:
                                props.update({'marker': '>', 'markersize': 2, 'markevery': 0.1})
                            ax.plot(avg.index, avg.values, **props)
        
        # Configure axis
        xlabel = configure_axis(ax, t_max, scale_type, i % cols == 0)  # ylabel only on leftmost
        ax.set_title(key)
        
        if i < n - cols:  # not bottom row
            ax.set_xlabel("")
            if scale_type == 'linear':
                ax.xaxis.offsetText.set_visible(False)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    return fig

if __name__ == "__main__":
    set_plot_style()
    
    files = [f for f in os.listdir("../../Output") if f.endswith(".csv") and "default_run_avg" in f]
    if not files:
        print("No files found")
        exit()
    
    for i, f in enumerate(files):
        print(f"{i}: {f}")
    
    idx = int(input("File index: "))
    scale = input("Scale type (linear/log/symlog): ").lower().strip()
    if scale not in ['linear', 'log', 'symlog']:
        scale = 'linear'
    t_max = 55000
    
    data = pd.read_csv(f"../../Output/{files[idx]}")
    processed = process_data(data, t_max, scale)
    
    N, n = files[idx].split("_")[4], files[idx].split("_")[6]
    suffix = f"_{scale}" if scale != 'linear' else ""
    today = date.today()
    
    plot_network_dynamics(processed, t_max, scale,
                         f"../../Figs/Trajectories/network_dynamics_N{N}_n_{n}_{today}{suffix}.pdf")
    
    plot_single_topology_dynamics(processed, t_max, scale,
                                 f"../../Figs/Trajectories/single_topology_N{N}_n_{n}_{today}{suffix}.pdf")