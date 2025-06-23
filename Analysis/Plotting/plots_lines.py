import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, ScalarFormatter, AutoMinorLocator
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec

# Line width parameters
line_params = {
    "data_line_width": 0.5,
    "axis_line_width": 0.8,
    "grid_line_width": 0.5,
    "tick_major_width": 0.8,
    "tick_minor_width": 0.6,
    "markersize": 3
}

cm = 1/2.54
FONT_SIZE = 7
SAVE_SIZE = (17.8*cm, 8.9*cm)

def set_plot_style():
    sns.set_style("white")
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'axes.linewidth': line_params["axis_line_width"],
        'lines.linewidth': 1.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (17.8*cm, 8.9*cm),
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'mathtext.default': 'regular',
        'axes.formatter.use_mathtext': True,
        'axes.axisbelow': True
    })

def process_data(data, t_max, log_time=False):
    required_columns = ['t', 'avg_state', 'std_states', 'scenario', 'rewiring', 'type']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in the data.")

    data = data[data['t'] <= t_max].copy()
    
    # Filter out t=0 for log scale (can't take log of 0)
    if log_time:
        data = data[data['t'] > 0].copy()
    
    data['rewiring'] = data['rewiring'].fillna('none')
    data['scenario'] = data['scenario'].fillna('none')
    data['scenario_grouped'] = data['scenario'].str.cat(data['rewiring'], sep='_')
    data = data.drop(columns=['scenario', 'rewiring']).rename(columns={'std_states': 'polarization'})

    return pd.melt(data, id_vars=['t', 'type', 'scenario_grouped'], 
                   value_vars=['avg_state', 'polarization'], 
                   var_name='measure', value_name='value')

# Global color scheme
PLOT_COLORS = {
    'none_none': '#EE7733', 'random_none': '#0077BB', 'biased_same': '#33BBEE',
    'biased_diff': '#009988', 'bridge_same': '#CC3311', 'bridge_diff': '#EE3377',
    'wtf_none': '#BBBBBB', 'node2vec_none': '#44BB99'
}

NETWORK_DISPLAY_NAMES = {'cl': 'CSF', 'DPAH': 'DPAH', 'Twitter': 'Twitter', 'FB': 'FB'}

def configure_axis_style(ax, t_max, log_time=False, data=None):
    ax.set_ylim(-0.6, 1.1)
    
    if log_time:
        ax.set_xscale('log')
        # Get actual min/max from filtered data if available
        if data is not None:
            t_min = data['t'].min() if len(data) > 0 else 1
            t_max_actual = data['t'].max() if len(data) > 0 else t_max
        else:
            t_min = 1
            t_max_actual = t_max
        ax.set_xlim(t_min, t_max_actual)
        ax.set_xlabel("Time, t (log scale)")
    else:
        ax.set_xlim(0, t_max)
        sci_formatter = ScalarFormatter(useMathText=True)
        sci_formatter.set_scientific(True)
        sci_formatter.set_powerlimits((-2, 1))
        sci_formatter._precision = 2
        ax.xaxis.set_major_formatter(sci_formatter)
        ax.set_xticks([0, 10000, 20000, 30000, 40000])
        ax.set_xlabel("Time, t")
    
    ax.grid(True, alpha=0.4, linestyle='--', which='major', zorder=5)
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(line_params["axis_line_width"])
        spine.set_zorder(100)

    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    
    if not log_time:
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    for axis in ['x', 'y']:
        ax.tick_params(axis=axis, reset=True)
        ax.tick_params(axis=axis, which='major', direction='out', length=2, 
                      width=line_params["tick_major_width"], colors='black', zorder=100,
                      bottom=True, top=False, left=True, right=False,
                      labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax.tick_params(axis=axis, which='minor', direction='out', length=1,
                      width=line_params["tick_minor_width"], colors='black', zorder=100,
                      bottom=True, top=True, left=True, right=True,
                      labelbottom=False, labeltop=False, labelleft=False, labelright=False)

def plot_network_dynamics(data, t_max=50, log_time=False, output_file=None):
    scenario_color_map = {scenario: PLOT_COLORS.get('_'.join(scenario.split('_')[:2]).lower(), '#FE6900') 
                         for scenario in data['scenario_grouped'].unique()}

    avg_state_data = data[data['measure'] == 'avg_state'].copy()
    polarization_data = data[data['measure'] == 'polarization'].copy()

    g = sns.relplot(data=avg_state_data, x='t', y='value', hue='scenario_grouped',
                   col='type', linewidth=line_params["data_line_width"], linestyle='-',
                   kind='line', col_wrap=2, height=4, aspect=1, palette=scenario_color_map, legend=False)

    g.fig.set_size_inches(11.8*cm, 11*cm)
    
    # Add polarization data with dashed lines
    for ax_idx, ax in enumerate(g.axes.flat):
        network_type = list(avg_state_data['type'].unique())[ax_idx]
        
        for scenario in polarization_data['scenario_grouped'].unique():
            pol_data = polarization_data[(polarization_data['scenario_grouped'] == scenario) & 
                                       (polarization_data['type'] == network_type)]
            
            if not pol_data.empty:
                ax.plot(pol_data['t'], pol_data['value'], linestyle='--', dashes=(4, 2),
                       color=scenario_color_map[scenario], linewidth=line_params["data_line_width"])

    ylabel = "Cooperativity, ⟨a⟩"
    xlabel = "Time, t (log scale)" if log_time else "$Time, t$"
    g.set_axis_labels(xlabel, ylabel)
    
    for ax, title in zip(g.axes.flat, [NETWORK_DISPLAY_NAMES.get(network, network) 
                                     for network in avg_state_data['type'].unique()]):
        ax.set_title(title)
        configure_axis_style(ax, t_max, log_time, data)
        
    for i, ax in enumerate(g.axes.flat):
        is_bottom_row = i >= 2
        if not is_bottom_row:
            ax.set_xlabel("")
            if not log_time:
                ax.xaxis.offsetText.set_visible(False)
    
    g.fig.subplots_adjust(top=0.84, bottom=0.20, hspace=0.45, wspace=0.28, left=0.1, right=0.95)
    
    # Add legends
    fig = g.fig
    legend_ax = fig.add_axes([0.15, 0.90, 0.7, 0.04])
    legend_ax.axis('off')
    
    line_elements = [Line2D([], [], color='black', linestyle='-', label='cooperativity'),
                    Line2D([], [], color='black', linestyle='--', dashes=(4, 2), label='polarization')]
    legend_ax.legend(handles=line_elements, ncol=2, loc='center', frameon=True, bbox_to_anchor=(0.5, 0.5))

    bottom_legend_ax = fig.add_axes([0.15, 0.03, 0.7, 0.05])
    bottom_legend_ax.axis('off')
    
    color_elements = [Line2D([], [], color=PLOT_COLORS[k], label=v) for k, v in {
        'none_none': 'static', 'random_none': 'random', 'biased_same': 'local (similar)',
        'biased_diff': 'local (opposite)', 'bridge_same': 'bridge (similar)', 'bridge_diff': 'bridge (opposite)'
    }.items()]
    
    if any('wtf' in s for s in data['scenario_grouped']):
        color_elements.append(Line2D([], [], color=PLOT_COLORS['wtf_none'], label='wtf'))
    if any('node2vec' in s for s in data['scenario_grouped']):
        color_elements.append(Line2D([], [], color=PLOT_COLORS['node2vec_none'], label='node2vec'))
    
    bottom_legend_ax.legend(handles=color_elements, ncol=4, loc='center', frameon=True, bbox_to_anchor=(0.5, 0.5))

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return g

def plot_single_topology_dynamics(data, t_max=50, log_time=False, output_file=None):
    scenario_categories = {'none_none': 'static', 'random_none': 'random', 'biased_same': 'local',    
                          'biased_diff': 'local', 'bridge_same': 'bridge', 'bridge_diff': 'bridge',  
                          'wtf_none': 'wtf', 'node2vec_none': 'node2vec'}
    
    scenario_to_color = {'static': 'none_none', 'random': 'random_none',
                        'local': {'same': 'biased_same', 'diff': 'biased_diff'},
                        'bridge': {'same': 'bridge_same', 'diff': 'bridge_diff'},
                        'wtf': 'wtf_none', 'node2vec': 'node2vec_none'}
    
    dpah_data = data[data['type'] == 'DPAH'].copy()
    cl_data = data[data['type'] == 'cl'].copy()
    
    plot_configs = {}
    for category, label in {'static': 'A', 'random': 'B', 'local': 'C', 
                          'bridge': 'D', 'wtf': 'E', 'node2vec': 'F'}.items():
        scenarios = [s for s in data['scenario_grouped'].unique() 
                    if scenario_categories.get('_'.join(s.split('_')[:2]).lower()) == category]
        if scenarios:
            plot_configs[label] = {'scenarios': scenarios}

    n_plots = len(plot_configs)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(17.8*cm, 12*cm))
    
    top, bottom, hspace, wspace, left, right = 0.90, 0.16, 0.27, 0.22, 0.1, 0.95
    plt.subplots_adjust(top=top, bottom=bottom, hspace=hspace, wspace=wspace, left=left, right=right)
    
    # Add legends
    legend_ax = fig.add_axes([left, 0.95, right-left, 0.05])
    legend_ax.axis('off')
    
    line_elements = [Line2D([], [], color='black', linestyle='-', label='cooperativity'),
                    Line2D([], [], color='black', linestyle='--', dashes=(4, 2), label='polarization'),
                    Line2D([], [], color='black', linestyle='-', marker='>', markersize=2, 
                           markevery=0.1, label='directed (DPAH)'),
                    Line2D([], [], color='black', linestyle='-', label='undirected (CSF)')]
    legend_ax.legend(handles=line_elements, ncol=5, loc='center', frameon=True, 
                    bbox_to_anchor=(0.5, 0.5), fontsize=FONT_SIZE-2)

    bottom_legend_ax = fig.add_axes([left, 0.03, right-left, 0.01])
    bottom_legend_ax.axis('off')
    
    color_elements = [Line2D([], [], color=PLOT_COLORS[k], label=v) for k, v in {
        'none_none': 'static', 'random_none': 'random', 'biased_same': 'local (similar)',
        'biased_diff': 'local (opposite)', 'bridge_same': 'bridge (similar)', 'bridge_diff': 'bridge (opposite)',
        'wtf_none': 'wtf', 'node2vec_none': 'node2vec'
    }.items()]
    bottom_legend_ax.legend(handles=color_elements, ncol=4, loc='center', frameon=True, bbox_to_anchor=(0.5, 0.5))

    gs = GridSpec(n_rows, n_cols, figure=fig, top=top, bottom=bottom, hspace=hspace, wspace=wspace, left=left, right=right)
    axes = [fig.add_subplot(gs[i // n_cols, i % n_cols]) for i in range(min(n_rows * n_cols, n_plots))]
    
    # Get static data for both networks
    static_dpah = dpah_data[dpah_data['scenario_grouped'].str.lower().str.startswith('none_none')].copy()
    static_cl = cl_data[cl_data['scenario_grouped'].str.lower().str.startswith('none_none')].copy()
    
    # Filter out t=0 for log scale in static data too
    if log_time:
        static_dpah = static_dpah[static_dpah['t'] > 0].copy()
        static_cl = static_cl[static_cl['t'] > 0].copy()
    
    for idx, (key, config) in enumerate(plot_configs.items()):
        ax = axes[idx]
        is_static = key == 'A'
        
        for measure in ['avg_state', 'polarization']:
            # Plot static references
            for data_type, static_data, is_directed in [(dpah_data, static_dpah, True), (cl_data, static_cl, False)]:
                static_measure = static_data[static_data['measure'] == measure]
                if not static_measure.empty:
                    static_avg = static_measure.groupby('t')['value'].mean()
                    
                    line_props = {'color': PLOT_COLORS['none_none'],
                                 'linestyle': '-' if measure == 'avg_state' else '--',
                                 'linewidth': line_params["data_line_width"], 'alpha': 0.7 if not is_static else 1.0}
                    
                    if measure == 'polarization':
                        line_props['dashes'] = (4, 2)
                        
                    if is_directed:
                        line_props.update({'marker': '>', 'markersize': 2,
                                         'markevery': 0.1 if measure == 'avg_state' else 0.15})
                    
                    ax.plot(static_avg.index, static_avg.values, **line_props)
            
            # Plot scenario data
            if not is_static:
                for scenario in config['scenarios']:
                    base_scenario = '_'.join(scenario.split('_')[:2]).lower()
                    scenario_type = scenario_categories.get(base_scenario)
                    
                    if scenario_type in scenario_to_color:
                        if scenario_type in ['local', 'bridge']:
                            sub_type = scenario.split('_')[-1].lower()
                            color_key = scenario_to_color[scenario_type][sub_type]
                        else:
                            color_key = scenario_to_color[scenario_type]
                            
                        for data_type, is_directed in [(dpah_data, True), (cl_data, False)]:
                            scenario_data = data_type[data_type['scenario_grouped'] == scenario]
                            scenario_measure = scenario_data[scenario_data['measure'] == measure]
                            
                            if not scenario_measure.empty:
                                scenario_avg = scenario_measure.groupby('t')['value'].mean()
                                
                                line_props = {'color': PLOT_COLORS[color_key],
                                             'linestyle': '-' if measure == 'avg_state' else '--',
                                             'linewidth': line_params["data_line_width"]}
                                
                                if measure == 'polarization':
                                    line_props['dashes'] = (4, 2)
                                    
                                if is_directed:
                                    line_props.update({'marker': '>', 'markersize': 2,
                                                     'markevery': 0.1 if measure == 'avg_state' else 0.15})
                                
                                ax.plot(scenario_avg.index, scenario_avg.values, **line_props)
        
        is_bottom_row = idx >= (n_plots - n_cols)
        
        if log_time:
            xlabel = "Time, t (log scale)"
        else:
            xlabel = "Time, t"
            
        ax.set_ylim(-0.6, 1.1)
        if is_bottom_row:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel("")
            
        if idx % n_cols == 0:
            ax.set_ylabel(r'Cooperativity, $\langle a \rangle$')
        else:
            ax.set_ylabel('')
            
        ax.set_title(f'{key}')
        configure_axis_style(ax, t_max, log_time, data)
        
        if not is_bottom_row and not log_time:
            ax.xaxis.offsetText.set_visible(False)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig

# Main execution
if __name__ == "__main__":
    set_plot_style()
    
    file_list = [f for f in os.listdir("../../Output") if f.endswith(".csv") and "default_run_avg" in f]
    if not file_list:
        print("No suitable files found in the Output directory.")
        exit()

    for i, file in enumerate(file_list):
        print(f"{i}: {file}")

    file_index = int(input("Enter the index of the file you want to plot: "))
    if file_index < 0 or file_index >= len(file_list):
        print("Invalid file index.")
        exit()

    # Ask for log time option
    log_time_input = input("Use log time scale? (y/n): ").lower().strip()
    log_time = log_time_input in ['y', 'yes', '1', 'true']

    data = pd.read_csv(os.path.join("../../Output", file_list[file_index]))
    t_max = 55
    get_N, get_n = file_list[file_index].split("_")[4], file_list[file_index].split("_")[6]
    
    processed_data = process_data(data, t_max, log_time)
    
    today = date.today()
    log_suffix = "_logtime" if log_time else ""
    
    plot_network_dynamics(processed_data, t_max, log_time,
                         f"../../Figs/Trajectories/network_dynamics_comparison_N{get_N}_n{get_n}_{today}{log_suffix}.pdf")
    
    plot_single_topology_dynamics(processed_data, t_max, log_time,
                                 f"../../Figs/Trajectories/single_topology_dynamics_comparison_N{get_N}_n{get_n}_{today}{log_suffix}.pdf")