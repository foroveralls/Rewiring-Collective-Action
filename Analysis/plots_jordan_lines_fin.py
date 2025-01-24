import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import AutoMinorLocator

params = {"line_width": 1.1}

def set_plot_style():
    """Set consistent style elements for all plots"""
    sns.set_style("white")
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.linewidth': 1.5,
        'lines.linewidth': 1.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out'
    })



def process_data(data, t_max):
    """Process and prepare data for plotting"""
    # Ensure all required columns are present
    required_columns = ['t', 'avg_state', 'std_states', 'scenario', 'rewiring', 'type']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in the data.")

    # Filter data based on t_max
    data = data[data['t'] <= t_max]

    # Create the scenario_grouped column
    data['rewiring'] = data['rewiring'].fillna('none')
    data['scenario'] = data['scenario'].fillna('none')
    data['scenario_grouped'] = data['scenario'].str.cat(data['rewiring'], sep='_')
    data = data.drop(columns=['scenario', 'rewiring'])
    
    # Rename 'std_states' to 'polarization' for consistency
    data = data.rename(columns={'std_states': 'polarization'})

    # Melt the dataframe to long format for easier plotting
    id_vars = ['t', 'type', 'scenario_grouped']
    value_vars = ['avg_state', 'polarization']
    data_long = pd.melt(data, id_vars=id_vars, value_vars=value_vars, 
                       var_name='measure', value_name='value')

    return data_long

def process_scenario_name(scenario_tuple):
    """Convert scenario tuple to standardized string format"""
    scenario, sub_scenario, network_type = scenario_tuple
    scenario = scenario.lower()
    sub_scenario = sub_scenario.lower() if sub_scenario != 'None' else 'none'
    return f"{scenario}_{sub_scenario}"


# Global color scheme using colorblind-friendly colors
PLOT_COLORS = {
    'none_none': '#EE7733',    # Orange
    'random_none': '#0077BB',  # Blue
    'biased_same': '#33BBEE',  # Cyan
    'biased_diff': '#009988',  # Teal
    'bridge_same': '#CC3311',  # Red
    'bridge_diff': '#EE3377',  # Magenta
    'wtf_none': '#BBBBBB',     # Grey
    'node2vec_none': '#44BB99' # Blue-green
}



def plot_network_dynamics(data, t_max=50, output_file=None):
    """Plot network dynamics across different network types"""
    # Create a mapping for colors based on scenario_grouped and friendly names
    scenario_color_map = {}
    friendly_names = {
        'none_none': 'static',
        'random_none': 'random',
        'biased_same': 'local (similar)',
        'biased_diff': 'local (opposite)',
        'bridge_same': 'bridge (similar)',
        'bridge_diff': 'bridge (opposite)',
        'wtf_none': 'wtf',
        'node2vec_none': 'node2vec'
    }
    
    # Map the actual scenario names to colors
    for scenario in data['scenario_grouped'].unique():
        base_scenario = '_'.join(scenario.split('_')[:2]).lower()
        scenario_color_map[scenario] = PLOT_COLORS.get(base_scenario, '#FE6900')

    # Create plot with 2x2 layout
    g = sns.relplot(
        data=data,
        x='t', y='value',
        hue='scenario_grouped',
        col='type',
        style='measure',
        linewidth= params["line_width"],
        kind='line',
        col_wrap=2,  # Force 2x2 layout
        height=4, aspect=1,
        palette=scenario_color_map,
        legend=False  # Remove default legend
    )

    g.set_axis_labels("Time [timestep / system size]", "Value")
    g.set_titles("{col_name}")

    # Apply consistent styling to all subplot
    for ax in g.axes.flat:
        ax.set_ylim(-0.6, 1.1)
        ax.set_xlim(0, t_max)
        ax.grid(True, alpha=0.4, linestyle='--', which='both')
        ax.set_axisbelow(True)
        
        # Update line styles for polarization
        for line in ax.lines:
            if 'polarization' in line.get_label():
                line.set_linestyle('--')
        
        # Set spine width
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # Set ticks
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(direction='out', length=5, width=1.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xscale('linear')

    # Create measure type legend elements
    measure_elements = [
        Line2D([0], [0], color='black', linestyle='-', label='cooperativity'),
        Line2D([0], [0], color='black', linestyle='--', label='polarization')
    ]
    
    # Create scenario legend elements
    scenario_elements = []
    for scenario_name, color in PLOT_COLORS.items():
        if scenario_name in friendly_names:
            scenario_elements.append(
                Line2D([0], [0], color=color, label=friendly_names[scenario_name])
            )

     # Add measure legend at the top right - adjusted position
    g.fig.legend(handles=measure_elements,
                title='Measures',
                bbox_to_anchor=(0.9, 0.9),  # Moved closer to plot
                loc='upper left',
                frameon=True)

   # Add scenarios legend below measures - adjusted position
    g.fig.legend(handles=scenario_elements,
                title='Scenarios',
                bbox_to_anchor=(0.9, 0.5),  # Moved closer to plot
                loc='center left',
                frameon=True)

    # Adjust layout to make room for legend
    g.fig.tight_layout()
    plt.subplots_adjust(right=0.88)  # Make room for the legends
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return g

def plot_single_topology_dynamics(data, t_max=50, output_file=None):
    """Plot single topology dynamics with proper static references and simplified titles"""
    # Setup configurations
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
    
    # Setup correct scenario mappings for colors
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
    
    # Plot configs (A-F panels)
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
    fig = plt.figure(figsize=(6*n_cols, 5*n_rows + 1.5))
    
    # Add combined line style and network type legend at top
    legend_ax = fig.add_axes([0.15, 0.95, 0.7, 0.02])
    legend_ax.axis('off')
    
    # Line style legend elements
    line_style_elements = [
        Line2D([], [], color='black', linestyle='-', label='cooperativity'),
        Line2D([], [], color='black', linestyle='--', label='polarization'),
        Line2D([], [], color='black', linestyle='-', marker='>', markersize=5, 
               markevery=0.1, label='directed (DPAH)'),
        Line2D([], [], color='black', linestyle='-', label='undirected (cl)')
    ]
    legend_ax.legend(handles=line_style_elements, ncol=4, loc='center', 
                    frameon=True, bbox_to_anchor=(0.5, 0.5))

    # Add bottom legend for scenario colors
    bottom_legend_ax = fig.add_axes([0.15, 0.005, 0.7, 0.02])
    bottom_legend_ax.axis('off')
    
    # Color legend elements - using friendly names
    color_elements = [
        Line2D([], [], color=PLOT_COLORS['none_none'], label='static'),
        Line2D([], [], color=PLOT_COLORS['random_none'], label='random'),
        Line2D([], [], color=PLOT_COLORS['biased_same'], label='local (similar)'),
        Line2D([], [], color=PLOT_COLORS['biased_diff'], label='local (opposite'),
        Line2D([], [], color=PLOT_COLORS['bridge_same'], label='bridge (similar)'),
        Line2D([], [], color=PLOT_COLORS['bridge_diff'], label='bridge (opposite)'),
        Line2D([], [], color=PLOT_COLORS['wtf_none'], label='wtf'),
        Line2D([], [], color=PLOT_COLORS['node2vec_none'], label='node2vec')
    ]
    bottom_legend_ax.legend(handles=color_elements, ncol=4, loc='center', 
                          frameon=True, bbox_to_anchor=(0.5, 0.5))

    # Create main subplot grid
    gs = plt.GridSpec(n_rows, n_cols, figure=fig, top=0.92, bottom=0.1)
    axes = [fig.add_subplot(gs[i // n_cols, i % n_cols]) 
            for i in range(min(n_rows * n_cols, n_plots))]
    
    # Get static network data for both network types
    static_dpah = dpah_data[dpah_data['scenario_grouped'].str.lower().str.startswith('none_none')].copy()
    static_cl = cl_data[cl_data['scenario_grouped'].str.lower().str.startswith('none_none')].copy()
    
    # Plot each subplot
    for idx, (key, config) in enumerate(plot_configs.items()):
        ax = axes[idx]
        is_static = key == 'A'
        
        # Plot for both network types
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
                        'linewidth': params["line_width"],
                        'alpha': 0.7 if not is_static else 1.0
                    }
                    if is_directed:
                        line_props.update({
                            'marker': '>',
                            'markersize': 5,
                            'markevery': 0.1
                        })
                    
                    ax.plot(static_avg.index, static_avg.values, **line_props)
            
           # In the plotting section, modify the color selection part:
            if not is_static:
                for scenario in config['scenarios']:
                    base_scenario = '_'.join(scenario.split('_')[:2]).lower()
                    scenario_type = scenario_categories.get(base_scenario)
                    
                    if scenario_type in scenario_to_color:
                        # Get the correct color key based on scenario type
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
                                    'linewidth': params["line_width"]
                                }
                                if is_directed:
                                    line_props.update({
                                        'marker': '>',
                                        'markersize': 5,
                                        'markevery': 0.1
                                    })
                                
                                ax.plot(scenario_avg.index, scenario_avg.values, **line_props)
        
        # Customize subplot
        ax.set(xlim=(0, t_max), ylim=(-0.6, 1.1),
              xlabel="Time [timestep / system size]",
              ylabel='Value' if idx % n_cols == 0 else '',
              title=f'{key}')  # Simplified title
        
        ax.grid(True, alpha=0.4, linestyle='--', which='both')
        ax.set_axisbelow(True)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            
        ax.tick_params(direction='out', length=5, width=1.5)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xscale('linear')

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig


# Main execution
if __name__ == "__main__":
    # Set the unified style at the start
    set_plot_style()
    
    # Get list of relevant output files
    file_list = [f for f in os.listdir("../Output") if f.endswith(".csv") and "default_run_avg" in f]

    if not file_list:
        print("No suitable files found in the Output directory.")
        exit()

    # Print file list with indices
    for i, file in enumerate(file_list):
        print(f"{i}: {file}")

    # Get user input for file selection
    file_index = int(input("Enter the index of the file you want to plot: "))

    if file_index < 0 or file_index >= len(file_list):
        print("Invalid file index.")
        exit()

    # Load and process the data
    data = pd.read_csv(os.path.join("../Output", file_list[file_index]))
    t_max = 35000  # Adjusted to match reference plot
    get_N, get_n = file_list[file_index].split("_")[4], file_list[file_index].split("_")[6]
    
    # Process the data
    processed_data = process_data(data, t_max)
    
    # Generate plots
    today = date.today()
    plot_network_dynamics(processed_data, t_max, 
                         output_file=f"../Figs/Trajectories/network_dynamics_comparison_N{get_N}_n{get_n}_{today}.pdf")
    plot_single_topology_dynamics(processed_data, t_max, 
                                output_file=f"../Figs/Trajectories/single_topology_dynamics_comparison_N{get_N}_n{get_n}_{today}.pdf")