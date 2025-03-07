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

# Line width parameters - organized by element type for clarity
line_params = {
    "data_line_width": 0.7,   # Width of actual data lines
    "axis_line_width": 1.0,   # Width of axis spines
    "grid_line_width": 0.5,   # Width of grid lines
    "tick_major_width": 1.2,  # Width of major tick marks
    "tick_minor_width": 0.8,  # Width of minor tick marks
    "markersize" : 3
    
}

cm = 1/2.54

FONT_SIZE = 7
SAVE_SIZE = (17.8*cm, 8.9*cm)
def set_plot_style():
    """Set consistent style elements for all plots"""
    sns.set_style("white")
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'axes.linewidth': line_params["axis_line_width"],  # Control axis border thickness
        'lines.linewidth': 1.5,  # Default line width (overridden for data lines)
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (17.8*cm, 8.9*cm),
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'mathtext.default': 'regular',
        'axes.formatter.use_mathtext': True,
        'axes.axisbelow': True  # Make sure grid is below everything
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

# Mapping of network type codes to display names
NETWORK_DISPLAY_NAMES = {
    'cl': 'CSF',
    'DPAH': 'DPAH',
    'Twitter': 'Twitter',
    'FB': 'FB'
}

def ensure_grid_visibility(ax):
    """Apply aggressive grid visibility settings to an axis"""
    # Clear any existing grid
    ax.grid(False)
    
    # Set the background color to very light gray to make the grid more visible
    ax.set_facecolor('white')
    
    # Draw a new grid with strong visibility settings
    ax.grid(
        True,
        which='major',
        color='gray',     # Specific color
        linestyle='--',
        linewidth=line_params["grid_line_width"],    # Use parameter for grid line width
        alpha=0.3,        # Higher alpha
        zorder=1          # Very low zorder to ensure it's behind everything
    )
    
    # Ensure the grid is behind other elements
    ax.set_axisbelow(True)
    
    # After setting the grid, we need to make sure data and spines have higher zorder
    for line in ax.get_lines():
        line.set_zorder(10)  # Ensure lines are above grid
    
    # Make spines very prominent
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_zorder(100)  # Highest zorder
        spine.set_linewidth(line_params["axis_line_width"])  # Use parameter for spine width

def apply_grid_fix(plot_obj, output_file=None):
    """Apply grid fixes to either a relplot or a figure with axes"""
    if hasattr(plot_obj, 'axes') and hasattr(plot_obj, 'fig'):  # For relplot objects
        for ax in plot_obj.axes.flat:
            ensure_grid_visibility(ax)
        fig = plot_obj.fig
    else:  # For regular figure objects
        for ax in plot_obj.axes:
            if hasattr(ax, 'get_lines') and len(ax.get_lines()) > 0:  # Only apply to actual plot axes
                ensure_grid_visibility(ax)
        fig = plot_obj
    
    # Now save the plot if an output file was specified
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return plot_obj

def apply_grid_fix_to_network_dynamics(data, t_max=50, output_file=None):
    # Create the plot using the existing function
    g = plot_network_dynamics(data, t_max, None)  # Don't save yet
    return apply_grid_fix(g, output_file)

def apply_grid_fix_to_single_topology(data, t_max=50, output_file=None):
    # Create the plot using the existing function
    fig = plot_single_topology_dynamics(data, t_max, None)  # Don't save yet
    return apply_grid_fix(fig, output_file)

def configure_axis_style(ax, t_max):
    """Apply common axis styling configuration"""
    ax.set_ylim(-0.6, 1.1)
    ax.set_xlim(0, t_max)
    
    # Make sure grid is drawn and visible
    ax.grid(True, alpha=0.4, linestyle='--', which='major', zorder=5)
    ax.set_axisbelow(True)
    
    # Set spine width and ensure they're on top
    for spine in ax.spines.values():
        spine.set_visible(True) 
        spine.set_linewidth(line_params["axis_line_width"])  # Use parameter
        spine.set_zorder(100)  # Make sure spines are on top

    # Set x-ticks with scientific notation (×10^n format)
    sci_formatter = ScalarFormatter(useMathText=True)
    sci_formatter.set_scientific(True)
    sci_formatter.set_powerlimits((-2, 1))  # Force scientific notation
    sci_formatter._precision = 2  # Set precision directly (more digits)
    ax.xaxis.set_major_formatter(sci_formatter)
    ax.set_xticks([0, 10000, 20000, 30000, 40000])
    
    # Set y-ticks
    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
    
    # NEW CODE: Add formatter for y-axis to display only one decimal place
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    
    # Add minor tick locators
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    # Reset and set tick parameters
    for axis in ['x', 'y']:
        # Reset existing tick parameters
        ax.tick_params(axis=axis, reset=True)
        
        # Set major ticks with higher zorder
        ax.tick_params(
            axis=axis, 
            which='major', 
            direction='out', 
            length=5, 
            width=line_params["tick_major_width"],  # Use parameter
            colors='black',
            zorder=100,  # High zorder to ensure visibility
            bottom=True, top=False, left=True, right=False,  # Explicit positioning
            labelbottom=True, labeltop=False, labelleft=True, labelright=False
        )
        
        # Set minor ticks with higher zorder
        ax.tick_params(
            axis=axis, 
            which='minor', 
            direction='out', 
            length=3, 
            width=line_params["tick_minor_width"],  # Use parameter
            colors='black',
            zorder=100,  # High zorder to ensure visibility
            bottom=True, top=False, left=True, right=False,  # Explicit positioning
            labelbottom=False, labeltop=False, labelleft=False, labelright=False
        )

def add_markers_for_directed(line, is_directed, measure='avg_state'):
    """Set line styles without adding markers"""
    # Only set the line style based on measure, ignore is_directed
    if measure == 'polarization' or line.get_linestyle() == '--':
        line.set_linestyle('--')
        line.set_dashes((4, 2))
    else:
        line.set_linestyle('-')  # Explicitly ensure cooperativity lines are solid

def plot_network_dynamics(data, t_max=50, output_file=None):
    """Plot network dynamics across different network types with improved styling"""
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

    # Separate data for directed and undirected networks
    directed_networks = ['DPAH', 'Twitter']
    undirected_networks = ['cl', 'FB']
    
    # Create separate dataframes for avg_state and polarization
    avg_state_data = data[data['measure'] == 'avg_state'].copy()
    polarization_data = data[data['measure'] == 'polarization'].copy()

    # First get the standard figure size from rcParams
    width, height = plt.rcParams['figure.figsize']
    
    # Create plot with 2x2 layout for avg_state - explicitly set linestyle to solid
    g = sns.relplot(
        data=avg_state_data,
        x='t', y='value',
        hue='scenario_grouped',
        col='type',
        linewidth=line_params["data_line_width"],
        linestyle='-',  # Force solid lines for all cooperativity
        kind='line',
        col_wrap=2,  # Force a 2x2 layout
        height=4, aspect=1,
        palette=scenario_color_map,
        legend=False  # Remove default legend
    )

    g.fig.set_size_inches(11.4*cm, 11.4*cm)

    # Add polarization data with dashed lines
    for ax_idx, ax in enumerate(g.axes.flat):
        # Get the network type for this subplot
        network_type = list(avg_state_data['type'].unique())[ax_idx]
        
        # Plot polarization data with dashed lines
        for scenario in polarization_data['scenario_grouped'].unique():
            pol_data = polarization_data[
                (polarization_data['scenario_grouped'] == scenario) & 
                (polarization_data['type'] == network_type)
            ]
            
            if not pol_data.empty:
                # Plot with dashed lines only (no markers)
                ax.plot(
                    pol_data['t'], 
                    pol_data['value'], 
                    linestyle='--',  # Always use dashed line for polarization
                    dashes=(4, 2),   # Explicit dash pattern
                    color=scenario_color_map[scenario],
                    linewidth=line_params["data_line_width"]  # Use parameter for data lines
                )

    g.set_axis_labels("$Time, t$", "Cooperativity, ⟨x⟩")
    for ax, title in zip(g.axes.flat, [NETWORK_DISPLAY_NAMES.get(network, network) 
                                 for network in avg_state_data['type'].unique()]):
        ax.set_title(title)
    
    # Apply consistent styling to all subplots
    for ax in g.axes.flat:
        configure_axis_style(ax, t_max)
    
    # Adjust spacing between subplots and figure edges - adjusted for better spacing
    g.fig.subplots_adjust(top=0.86, bottom=0.18, hspace=0.35, wspace=0.22, left=0.1, right=0.95)
    
    # After creating the relplot and setting up subplots
    for i, ax in enumerate(g.axes.flat):
        configure_axis_style(ax, t_max)
        
        # Determine if this is in the bottom rowcsv
        # For a 2x2 grid with col_wrap=2, axes 2 and 3 are the bottom row
        is_bottom_row = i >= 2  # Assuming 4 subplots with indices 0,1,2,3
        
        # Hide x-axis labels for all but bottom row
        if not is_bottom_row:
            ax.set_xlabel("")
            # Hide the scientific notation (×10⁴)
            ax.xaxis.offsetText.set_visible(False)
    
    # Final pass to ensure line styles are correct
    for ax in g.axes.flat:
        lines = ax.get_lines()
        num_scenarios = len(avg_state_data['scenario_grouped'].unique())
        
        # First set are cooperativity lines - ensure they're solid
        for i in range(min(num_scenarios, len(lines))):
            lines[i].set_linestyle('-')
        
        # Remaining lines are polarization - ensure they're dashed
        for i in range(min(num_scenarios, len(lines)), len(lines)):
            lines[i].set_linestyle('--')
            lines[i].set_dashes((4, 2))
            
    # Create combined figure legend at top with more space and distance from plot
    # Moved higher for better separation
    fig = g.fig
    legend_ax = fig.add_axes([0.15, 0.90, 0.7, 0.04])  # [left, bottom, width, height] - Moved up
    legend_ax.axis('off')
    
    # Line style legend elements - with simpler labels (no directed/undirected distinction)
    line_style_elements = [
        Line2D([], [], color='black', linestyle='-', label='cooperativity'),
        Line2D([], [], color='black', linestyle='--', dashes=(4, 2), label='polarization')
    ]
    # Use 2 columns for the simplified legend
    legend_ax.legend(handles=line_style_elements, ncol=2, loc='center', 
                    frameon=True, bbox_to_anchor=(0.5, 0.5),
                    handletextpad=0.5, columnspacing=0.5, fontsize = FONT_SIZE-1)

    # Add bottom legend for scenario colors - moved lower for better separation
    bottom_legend_ax = fig.add_axes([0.15, 0.03, 0.7, 0.05])  # [left, bottom, width, height] - Moved down
    bottom_legend_ax.axis('off')
    
    # Color legend elements - using friendly names
    color_elements = [
        Line2D([], [], color=PLOT_COLORS['none_none'], label='static'),
        Line2D([], [], color=PLOT_COLORS['random_none'], label='random'),
        Line2D([], [], color=PLOT_COLORS['biased_same'], label='local (similar)'),
        Line2D([], [], color=PLOT_COLORS['biased_diff'], label='local (opposite)'),
        Line2D([], [], color=PLOT_COLORS['bridge_same'], label='bridge (similar)'),
        Line2D([], [], color=PLOT_COLORS['bridge_diff'], label='bridge (opposite)'),
    ]
    
    # Add wtf and node2vec if they exist in the data
    if 'wtf_none' in scenario_color_map or any('wtf' in s for s in data['scenario_grouped']):
        color_elements.append(Line2D([], [], color=PLOT_COLORS['wtf_none'], label='wtf'))
    
    if 'node2vec_none' in scenario_color_map or any('node2vec' in s for s in data['scenario_grouped']):
        color_elements.append(Line2D([], [], color=PLOT_COLORS['node2vec_none'], label='node2vec'))
    
    bottom_legend_ax.legend(handles=color_elements, ncol=4, loc='center', 
                          frameon=True, bbox_to_anchor=(0.5, 0.5), fontsize = FONT_SIZE-1,
                          columnspacing = 0.5)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return g

def plot_single_topology_dynamics(data, t_max=50, output_file=None):
    """Plot single topology dynamics with proper static references and simplified titles with y-axis label Cooperativity"""
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
    fig = plt.figure(figsize=(17.8*cm, 11.2*cm))
    
    # Adjust spacing for the single topology dynamics plot - increased margins
    top, bottom, hspace, wspace, left, right = 0.90, 0.16, 0.27, 0.22, 0.1, 0.95
    
    plot_width = right - left
    
    plt.subplots_adjust(top=top, bottom=bottom, hspace=hspace, wspace=wspace, left=left, right=right)
    
    # Add combined line style and network type legend at top - moved higher
    legend_ax = fig.add_axes([left, 0.95, plot_width, 0.05])  # [left, bottom, width, height] - Moved up
    legend_ax.axis('off')
    
    # Line style legend elements - updated to include dashed line with marker
    line_style_elements = [
        Line2D([], [], color='black', linestyle='-', label='cooperativity'),
        Line2D([], [], color='black', linestyle='--', dashes=(4, 2), label='polarization'),
        Line2D([], [], color='black', linestyle='-', marker='>', markersize=2, 
               markevery=0.1, label='directed (DPAH)'),
        Line2D([], [], color='black', linestyle='-', label='undirected (CSF)')
    ]
    legend_ax.legend(handles=line_style_elements, ncol=5, loc='center', 
                    frameon=True, bbox_to_anchor=(0.5, 0.5),
                    fontsize=FONT_SIZE-1)

    # Add bottom legend for scenario colors - moved lower
    bottom_legend_ax = fig.add_axes([left, 0.03, plot_width, 0.01])  # [left, bottom, width, height] - Moved down
    bottom_legend_ax.axis('off')
    
    # Color legend elements - using friendly names
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

    # Create main subplot grid with adjusted spacing
    gs = plt.GridSpec(n_rows, n_cols, figure=fig,
                    top=top, bottom=bottom, hspace=hspace, wspace=wspace, left=left, right=right)
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
                        'linewidth': line_params["data_line_width"],  # Use parameter for data lines
                        'alpha': 0.7 if not is_static else 1.0
                    }
                    
                    # Only add dashes parameter for polarization (dashed lines)
                    if measure == 'polarization':
                        line_props['dashes'] = (4, 2)
                        
                    if is_directed:
                        # Add markers to both avg_state and polarization for directed networks
                        line_props.update({
                            'marker': '>',
                            'markersize': 2,
                            'markevery': 0.1 if measure == 'avg_state' else 0.15  # Slightly different spacing for polarization
                        })
                    
                    ax.plot(static_avg.index, static_avg.values, **line_props)
            
            # Plot scenario data
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
                                    'linewidth': line_params["data_line_width"]  # Use parameter for data lines
                                }
                                
                                # Only add dashes parameter for polarization (dashed lines)
                                if measure == 'polarization':
                                    line_props['dashes'] = (4, 2)
                                    
                                if is_directed:
                                    # Add markers to both avg_state and polarization for directed networks
                                    line_props.update({
                                        'marker': '>',
                                        'markersize': 2,
                                        'markevery': 0.1 if measure == 'avg_state' else 0.15  # Slightly different spacing for polarization
                                    })
                                
                                ax.plot(scenario_avg.index, scenario_avg.values, **line_props)
        
        # Determine if this is in the bottom row
        is_bottom_row = idx >= (n_plots - n_cols)
        
        # Customize subplot
        ax.set(xlim=(0, t_max), ylim=(-0.6, 1.1),
              # Only set xlabel for bottom row
              xlabel="$Time, t$" if is_bottom_row else "",
              ylabel='Cooperativity, ⟨x⟩' if idx % n_cols == 0 else '',
              title=f'{key}')
            
        # Apply common axis styling
        configure_axis_style(ax, t_max)
        
        
        # Hide scientific notation for non-bottom row
        if not is_bottom_row:
            ax.xaxis.offsetText.set_visible(False)
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
    t_max = 45000  # Adjusted to match reference plot
    get_N, get_n = file_list[file_index].split("_")[4], file_list[file_index].split("_")[6]
    
    # Process the data
    processed_data = process_data(data, t_max)
    
    # Generate plots
    today = date.today()
    apply_grid_fix_to_network_dynamics(processed_data, t_max, 
                       output_file=f"../Figs/Trajectories/network_dynamics_comparison_N{get_N}_n{get_n}_{today}.pdf")
    
    apply_grid_fix_to_single_topology(processed_data, t_max, 
                      output_file=f"../Figs/Trajectories/single_topology_dynamics_comparison_N{get_N}_n{get_n}_{today}.pdf")