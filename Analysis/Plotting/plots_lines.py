import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
from matplotlib.gridspec import GridSpec

# Line width parameters (increased for better visibility at 110% zoom)
line_params = {
    "data_line_width": 1.0,  # Increased from 0.8
    "axis_line_width": 1.2,  # Increased from 0.8
    "grid_line_width": 0.6,  # Increased from 0.5
    "tick_major_width": 1.2,  # Increased from 0.8
    "tick_minor_width": 0.8,  # Increased from 0.6
    "markersize": 4  # Increased from 3
}

cm = 1/2.54
FONT_SIZE = 11  # Increased from 7 for better readability

# Color scheme
PLOT_COLORS = {
    'none_none': '#EE7733',
    'random_none': '#0077BB',
    'biased_same': '#33BBEE',
    'biased_diff': '#009988',
    'bridge_same': '#CC3311',
    'bridge_diff': '#EE3377',
    'wtf_none': '#BBBBBB',
    'node2vec_none': '#44BB99'
}

NETWORK_DISPLAY_NAMES = {
    'cl': 'CSF',
    'DPAH': 'DPA',
    'Twitter': 'Twitter',
    'FB': 'FB'
}

def set_plot_style():
    """Set consistent style elements for all plots"""
    sns.set_style("white")
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'axes.linewidth': line_params["axis_line_width"],
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (17.8*cm, 8.9*cm),
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'mathtext.default': 'regular',
        'axes.formatter.use_mathtext': True,
        'axes.axisbelow': True
    })

def process_data(data, t_max, scale_type='linear'):
    """Process and prepare data for plotting"""
    req_cols = ['t', 'avg_state', 'std_states', 'scenario', 'rewiring', 'type']
    for col in req_cols:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found")
    
    data = data[data['t'] <= t_max].copy()
    
    data['rewiring'] = data['rewiring'].fillna('none')
    data['scenario'] = data['scenario'].fillna('none')
    data['scenario_grouped'] = data['scenario'].str.cat(data['rewiring'], sep='_')
    
    return pd.melt(data.drop(columns=['scenario', 'rewiring']).rename(columns={'std_states': 'polarization'}),
                   id_vars=['t', 'type', 'scenario_grouped'], 
                   value_vars=['avg_state', 'polarization'], 
                   var_name='measure', value_name='value')

def configure_axis_style(ax, t_max, scale_type='linear', show_ylabel=True, show_xlabel=True):
    """Apply comprehensive axis styling with proper ticks and labels"""
    ax.set_ylim(-0.6, 1.1)
    
    if scale_type == 'symlog':
        ax.set_xscale('symlog', linthresh=50000)
        ax.set_xlim(0, t_max)
        xlabel = "Time, t (symlog)"
    else:  # linear
        ax.set_xlim(0, t_max)
        sci_formatter = ScalarFormatter(useMathText=True)
        sci_formatter.set_scientific(True)
        sci_formatter.set_powerlimits((-2, 1))
        ax.xaxis.set_major_formatter(sci_formatter)
        
        # Dynamic x-ticks based on t_max
        step = 10000 if t_max <= 60000 else 20000 if t_max <= 120000 else 50000
        ticks = np.arange(0, t_max + step, step)
        ax.set_xticks(ticks[ticks <= t_max])
        xlabel = "Time, t"
    
    # Grid configuration
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=line_params["grid_line_width"], zorder=1)
    ax.set_axisbelow(True)
    
    # Spine configuration
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(line_params["axis_line_width"])
        spine.set_zorder(100)

    # Y-axis configuration
    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    
    # Minor tick locators
    if scale_type == 'linear':
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    # Tick parameters
    ax.tick_params(
        axis='both', 
        which='major', 
        direction='out', 
        length=3, 
        width=line_params["tick_major_width"],
        colors='black',
        zorder=100,
        bottom=True, top=False, left=True, right=False,
        labelbottom=show_xlabel, labeltop=False, 
        labelleft=show_ylabel, labelright=False
    )
    
    ax.tick_params(
        axis='both', 
        which='minor', 
        direction='out', 
        length=1.5, 
        width=line_params["tick_minor_width"],
        colors='black',
        zorder=100,
        bottom=True, top=False, left=True, right=False,
        labelbottom=False, labeltop=False, 
        labelleft=False, labelright=False
    )
    
    return xlabel

def plot_network_dynamics(data, t_max=50, scale_type='linear', output_file=None):
    """Plot network dynamics across different network types with improved styling"""
    # Create color mapping
    scenario_color_map = {}
    for scenario in data['scenario_grouped'].unique():
        base_scenario = '_'.join(scenario.split('_')[:2]).lower()
        scenario_color_map[scenario] = PLOT_COLORS.get(base_scenario, '#FE6900')

    # Create separate dataframes
    avg_state_data = data[data['measure'] == 'avg_state'].copy()
    polarization_data = data[data['measure'] == 'polarization'].copy()

    # Create plot
    g = sns.relplot(
        data=avg_state_data,
        x='t', y='value',
        hue='scenario_grouped',
        col='type',
        linewidth=line_params["data_line_width"],
        linestyle='-',
        kind='line',
        col_wrap=2,
        height=4, aspect=1,
        palette=scenario_color_map,
        legend=False
    )

    g.fig.set_size_inches(17.8*cm, 13*cm)
    
    # Add polarization lines
    for ax_idx, ax in enumerate(g.axes.flat):
        network_type = list(avg_state_data['type'].unique())[ax_idx]
        
        for scenario in polarization_data['scenario_grouped'].unique():
            pol_data = polarization_data[
                (polarization_data['scenario_grouped'] == scenario) & 
                (polarization_data['type'] == network_type)
            ]
            
            if not pol_data.empty:
                ax.plot(
                    pol_data['t'], 
                    pol_data['value'], 
                    linestyle='--',
                    dashes=(4, 2),
                    color=scenario_color_map[scenario],
                    linewidth=line_params["data_line_width"]
                )

    # Apply styling to each subplot
    xlabel = "Time, t"  # default
    for i, ax in enumerate(g.axes.flat):
        is_bottom_row = i >= 2
        is_left_col = i % 2 == 0
        
        xlabel = configure_axis_style(ax, t_max, scale_type,
                                    show_ylabel=is_left_col, 
                                    show_xlabel=is_bottom_row)
        
        # Set titles
        network_type = list(avg_state_data['type'].unique())[i]
        ax.set_title(NETWORK_DISPLAY_NAMES.get(network_type, network_type))
        
        # Only show scientific notation offset on bottom row
        if not is_bottom_row and scale_type == 'linear':
            ax.xaxis.offsetText.set_visible(False)

    # Set axis labels
    g.set_axis_labels(xlabel, r"Opinion, $\langle a \rangle$")
    
    # Adjust spacing
    g.fig.subplots_adjust(top=0.84, bottom=0.20, hspace=0.25, wspace=0.28, left=0.1, right=0.95)
    
    # Add legends
    fig = g.fig
    legend_ax = fig.add_axes([0.15, 0.88, 0.7, 0.04])
    legend_ax.axis('off')
    
    line_style_elements = [
        Line2D([], [], color='black', linestyle='-', label='opinion'),
        Line2D([], [], color='black', linestyle='--', dashes=(4, 2), label='polarization')
    ]
    legend_ax.legend(handles=line_style_elements, ncol=2, loc='center', 
                    frameon=True, bbox_to_anchor=(0.5, 0.5))

    bottom_legend_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
    bottom_legend_ax.axis('off')
    
    color_elements = [
        Line2D([], [], color=PLOT_COLORS['none_none'], label='static'),
        Line2D([], [], color=PLOT_COLORS['random_none'], label='random'),
        Line2D([], [], color=PLOT_COLORS['biased_same'], label='local (similar)'),
        Line2D([], [], color=PLOT_COLORS['biased_diff'], label='local (opposite)'),
        Line2D([], [], color=PLOT_COLORS['bridge_same'], label='bridge (similar)'),
        Line2D([], [], color=PLOT_COLORS['bridge_diff'], label='bridge (opposite)'),
    ]
    
    if any('wtf' in s for s in data['scenario_grouped']):
        color_elements.append(Line2D([], [], color=PLOT_COLORS['wtf_none'], label='wtf'))
    
    if any('node2vec' in s for s in data['scenario_grouped']):
        color_elements.append(Line2D([], [], color=PLOT_COLORS['node2vec_none'], label='node2vec'))
    
    bottom_legend_ax.legend(handles=color_elements, ncol=4, loc='center', 
                          frameon=True, bbox_to_anchor=(0.5, 0.5))

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return g

def plot_single_topology_dynamics(data, t_max=50, scale_type='linear', output_file=None):
    """Plot single topology dynamics with proper static references"""
    scenario_categories = {
        'none_none': 'static',
        'random_none': 'random',
        'biased_same': 'local',    
        'biased_diff': 'local',    
        'bridge_same': 'bridge',  
        'bridge_diff': 'bridge',  
        'wtf_none': 'wtf',
        'node2vec_none': 'node2vec'
    }
    
    scenario_to_color = {
        'static': 'none_none',
        'random': 'random_none',
        'local': {'same': 'biased_same', 'diff': 'biased_diff'},
        'bridge': {'same': 'bridge_same', 'diff': 'bridge_diff'},
        'wtf': 'wtf_none',
        'node2vec': 'node2vec_none'
    }
    
    # Split data by network type
    dpah_data = data[data['type'] == 'DPAH'].copy()
    cl_data = data[data['type'] == 'cl'].copy()
    
    # Plot configs
    plot_configs = {}
    for category, label in {'static': 'A', 'random': 'B', 'local': 'C', 
                          'bridge': 'D', 'wtf': 'E', 'node2vec': 'F'}.items():
        scenarios = []
        for scenario_group in data['scenario_grouped'].unique():
            base_scenario = '_'.join(scenario_group.split('_')[:2]).lower()
            if scenario_categories.get(base_scenario) == category:
                scenarios.append(scenario_group)
        if scenarios:
            plot_configs[label] = {'scenarios': scenarios}

    # Setup figure
    n_plots = len(plot_configs)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(17.8*cm, 12*cm))
    
    top, bottom, hspace, wspace, left, right = 0.90, 0.16, 0.27, 0.22, 0.1, 0.95
    plt.subplots_adjust(top=top, bottom=bottom, hspace=hspace, wspace=wspace, left=left, right=right)
    
    # Add legends
    plot_width = right - left
    
    legend_ax = fig.add_axes([left, 0.95, plot_width, 0.05])
    legend_ax.axis('off')
    
    line_style_elements = [
        Line2D([], [], color='black', linestyle='-', label='opinion'),
        Line2D([], [], color='black', linestyle='--', dashes=(4, 2), label='polarization'),
        Line2D([], [], color='black', linestyle='-', marker='>', markersize=2, 
               markevery=0.1, label='directed (DPA)'),
        Line2D([], [], color='black', linestyle='-', label='undirected (CSF)')
    ]
    legend_ax.legend(handles=line_style_elements, ncol=5, loc='center', 
                    frameon=True, bbox_to_anchor=(0.5, 0.5), fontsize=FONT_SIZE-2)

    bottom_legend_ax = fig.add_axes([left, 0.03, plot_width, 0.01])
    bottom_legend_ax.axis('off')
    
    color_elements = [
        Line2D([], [], color=PLOT_COLORS['none_none'], label='static'),
        Line2D([], [], color=PLOT_COLORS['random_none'], label='random'),
        Line2D([], [], color=PLOT_COLORS['biased_same'], label='local (similar)'),
        Line2D([], [], color=PLOT_COLORS['biased_diff'], label='local (opposite)'),
        Line2D([], [], color=PLOT_COLORS['bridge_same'], label='bridge (similar)'),
        Line2D([], [], color=PLOT_COLORS['bridge_diff'], label='bridge (opposite)'),
        Line2D([], [], color=PLOT_COLORS['wtf_none'], label='wtf'),
        Line2D([], [], color=PLOT_COLORS['node2vec_none'], label='node2vec')
    ]
    bottom_legend_ax.legend(handles=color_elements, ncol=4, loc='center', 
                          frameon=True, bbox_to_anchor=(0.5, 0.5))

    # Create subplot grid
    gs = GridSpec(n_rows, n_cols, figure=fig,
                 top=top, bottom=bottom, hspace=hspace, wspace=wspace, left=left, right=right)
    
    # Get static network data
    static_dpah = dpah_data[dpah_data['scenario_grouped'].str.lower().str.startswith('none_none')].copy()
    static_cl = cl_data[cl_data['scenario_grouped'].str.lower().str.startswith('none_none')].copy()
    
    # Plot each subplot
    for idx, (key, config) in enumerate(plot_configs.items()):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        is_static = key == 'A'
        
        # Plot for both network types and measures
        for measure in ['avg_state', 'polarization']:
            # Plot static references
            for data_type, static_data, is_directed in [
                (dpah_data, static_dpah, True),
                (cl_data, static_cl, False)
            ]:
                static_measure = static_data[static_data['measure'] == measure]
                if not static_measure.empty:
                    static_avg = static_measure.groupby('t')['value'].mean()
                    
                    line_props = {
                        'color': PLOT_COLORS['none_none'],
                        'linestyle': '-' if measure == 'avg_state' else '--',
                        'linewidth': line_params["data_line_width"],
                        'alpha': 0.7 if not is_static else 1.0
                    }
                    
                    if measure == 'polarization':
                        line_props['dashes'] = (4, 2)
                        
                    if is_directed:
                        line_props.update({
                            'marker': '>',
                            'markersize': 2,
                            'markevery': 0.1 if measure == 'avg_state' else 0.15
                        })
                    
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
                                
                                line_props = {
                                    'color': PLOT_COLORS[color_key],
                                    'linestyle': '-' if measure == 'avg_state' else '--',
                                    'linewidth': line_params["data_line_width"]
                                }
                                
                                if measure == 'polarization':
                                    line_props['dashes'] = (4, 2)
                                    
                                if is_directed:
                                    line_props.update({
                                        'marker': '>',
                                        'markersize': 2,
                                        'markevery': 0.1 if measure == 'avg_state' else 0.15
                                    })
                                
                                ax.plot(scenario_avg.index, scenario_avg.values, **line_props)
        
        # Configure axis
        is_bottom_row = idx >= (n_plots - n_cols)
        is_left_col = idx % n_cols == 0
        
        xlabel = configure_axis_style(ax, t_max, scale_type,
                                    show_ylabel=is_left_col, 
                                    show_xlabel=is_bottom_row)
        
        # Set labels and title
        ax.set_title(f'{key}')
        if is_left_col:
            ax.set_ylabel(r'Opinion, $\langle a \rangle$')
        if is_bottom_row:
            ax.set_xlabel(xlabel)
        
        # Hide scientific notation for non-bottom row
        if not is_bottom_row and scale_type == 'linear':
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

    scale = input("Scale type (linear/symlog): ").lower().strip()
    if scale not in ['linear', 'symlog']:
        scale = 'linear'

    data = pd.read_csv(os.path.join("../../Output", file_list[file_index]))
    t_max = 60000
    get_N, get_n = file_list[file_index].split("_")[4], file_list[file_index].split("_")[6]
    
    processed_data = process_data(data, t_max, scale)
    
    today = date.today()
    suffix = f"_{scale}" if scale != 'linear' else ""
    
    plot_network_dynamics(processed_data, t_max, scale,
                         output_file=f"../../Figs/Trajectories/network_dynamics_comparison_N{get_N}_n{get_n}_{today}{suffix}.pdf")
    
    plot_single_topology_dynamics(processed_data, t_max, scale,
                                output_file=f"../../Figs/Trajectories/single_topology_dynamics_comparison_N{get_N}_n{get_n}_{today}{suffix}.pdf")