import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import AutoMinorLocator

# Set unified style for all plots
def set_plot_style():
    """Set consistent style elements for all plots"""
    sns.set_style("white")
    plt.rcParams.update({
        # Font settings
        'font.family': 'Arial',
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        
        # Line and spine settings
        'axes.linewidth': 1.5,
        'lines.linewidth': 2,
        
        # Figure settings
        'figure.dpi': 300,
        'savefig.dpi': 300,
        
        # Grid settings
        'grid.alpha': 0,
        'grid.linestyle': '--',
        
        # Tick settings
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
    'none_none': '#648FFF',    # Blue for static network
    'random_none': '#785EF0',  # Purple
    'biased_same': '#DC267F',   # Magenta (darker)
    'biased_diff': '#FF69B0',   # Magenta (lighter)
    'bridge_same': '#FE6100',  # Orange (darker)
    'bridge_diff': '#FFB000',  # Orange (lighter)
    'wtf_none': '#24A608',     # Green
    'node2vec_none': '#FE6900'  # Different orange/red shade instead of black
}



def plot_network_dynamics(data, t_max=50, output_file=None):
    """Plot network dynamics across different network types"""
    # Create a mapping for colors based on scenario_grouped
    scenario_color_map = {}
    for scenario in data['scenario_grouped'].unique():
        base_scenario = '_'.join(scenario.split('_')[:2]).lower()
        scenario_color_map[scenario] = PLOT_COLORS.get(base_scenario, '#FE6900')

    g = sns.relplot(
        data=data,
        x='t', y='value',
        hue='scenario_grouped',
        col='type',
        style='measure',
        linewidth = 1.5,
        kind='line',
        facet_kws={'sharex': True, 'sharey': True},  # Force shared axes
        height=5, aspect=1.2,
        palette=scenario_color_map,
        legend='full'
    )

    g.set_axis_labels("Time [timestep / system size]", "Value")
    g.set_titles("{col_name} Network")
    g.fig.suptitle("Time evolution of cooperativity and polarization for various network types", 
                   fontsize=16, fontweight='bold', y=1.02)

    for ax in g.axes.flat:
        # Set consistent y-axis limits and grid
        ax.set_ylim(-0.6, 1.1)
        for line in ax.lines:
            if 'polarization' in line.get_label():
                line.set_linestyle('--')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # Set consistent x-axis limits and ticks
        ax.set_xlim(0, t_max)
        ax.grid(True, alpha=0.4, linestyle='--', which='both')
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
        
        # Set tick parameters
        ax.tick_params(direction='out', length=5, width=1.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())  # Add minor ticks
        ax.set_xscale('linear')  # Force linear scale
        
    g.tight_layout()
    g.fig.subplots_adjust(top=0.9, bottom=0.1)
    
    plt.legend(title='Rewiring Scenarios', loc='upper left')
    # g.add_legend(title="Rewiring Scenarios", 
    #              # Add this line
    #             bbox_to_anchor=(0.5, -0.1), 
    #             loc='upper center', 
    #             ncol=3)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def plot_single_topology_dynamics(data, t_max=1000, output_file=None):
    """
    Plot single topology dynamics with averaged trajectories for both directed and undirected networks.
    """
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
            if any(k == base_scenario and v == category for k, v in scenario_categories.items()):
                scenarios.append(scenario_group)
        if scenarios:
            plot_configs[label] = {'title': f"{category} rewiring" if category != 'static' 
                                          else "static network", 'scenarios': scenarios}

    # Setup figure
    n_plots = len(plot_configs)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(6*n_cols, 5*n_rows + 1.5))  # Added extra height for bottom legend
    
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
    bottom_legend_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
    bottom_legend_ax.axis('off')
    
    # Color legend elements
    color_elements = [
        Line2D([], [], color=PLOT_COLORS['none_none'], label='static'),
        Line2D([], [], color=PLOT_COLORS['random_none'], label='random'),
        Line2D([], [], color=PLOT_COLORS['biased_same'], label='local (same)'),
        Line2D([], [], color=PLOT_COLORS['biased_diff'], label='local (diff)'),
        Line2D([], [], color=PLOT_COLORS['bridge_same'], label='bridge (same)'),
        Line2D([], [], color=PLOT_COLORS['bridge_diff'], label='bridge (diff)'),
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
        scenario_type = config['title'].split()[0].lower()
        
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
                        'linewidth': 1.5,
                        'alpha': 0.7 if not is_static else 1.0
                    }
                    if is_directed:
                        line_props.update({
                            'marker': '>',
                            'markersize': 5,
                            'markevery': 0.1
                        })
                    
                    ax.plot(static_avg.index, static_avg.values, **line_props)
            
            # Plot scenario data if not static network
            if not is_static:
                for sub_type in ['same', 'diff'] if scenario_type in ['local', 'bridge'] else ['none']:
                    scenario_key = f"{scenario_type}_{sub_type}"
                    if scenario_type == 'local':
                        scenario_key = f"biased_{sub_type}"  # Special case for local/biased
                    
                    relevant_scenarios = [s for s in config['scenarios'] 
                                       if s.lower().startswith(scenario_key.lower())]
                    
                    if relevant_scenarios:
                        for data_type, is_directed in [(dpah_data, True), (cl_data, False)]:
                            scenario_data = data_type[data_type['scenario_grouped'].isin(relevant_scenarios)]
                            scenario_measure = scenario_data[scenario_data['measure'] == measure]
                            
                            if not scenario_measure.empty:
                                scenario_avg = scenario_measure.groupby('t')['value'].mean()
                                
                                line_props = {
                                    'color': PLOT_COLORS[scenario_key],
                                    'linestyle': '-' if measure == 'avg_state' else '--',
                                    'linewidth': 1.5,
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
              xlabel= "Time [timestep / system size]",  # Simplified x-label
              ylabel='Value' if idx % n_cols == 0 else '',
              title=f'{key} {config["title"]}')
        
        ax.grid(False)
        ax.grid(True, alpha=0.4, linestyle='--', which='both')
        ax.set_axisbelow(True)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            
        ax.tick_params(direction='out', length=5, width=1.5)
        
        # Simplified x-axis ticks
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        #ax.set_xticks(np.linspace(0, t_max, 6))
        ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.set_minor_locator(AutoMinorLocator())  # Add minor ticks
        ax.set_xscale('linear')  # Force linear scale
        #ax.set_xlabel("Time [timestep / system size]") 

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

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
    t_max = 30000  # Adjusted to match reference plot
    get_N = file_list[file_index].split("_")[5]
    
    # Process the data
    processed_data = process_data(data, t_max)
    
    # Generate plots
    today = date.today()
    plot_network_dynamics(processed_data, t_max, 
                         output_file=f"../Figs/Trajectories/network_dynamics_comparison_{get_N}_{today}.pdf")
    plot_single_topology_dynamics(processed_data, t_max, 
                                output_file=f"../Figs/Trajectories/single_topology_dynamics_comparison_{get_N}_{today}.pdf")