import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from matplotlib.ticker import FuncFormatter


#%% 
def process_data(data, t_max):
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
    data= data.drop(columns=['scenario', 'rewiring'])
    
    # Rename 'std_states' to 'polarization' for consistency with the plotting functions
    data = data.rename(columns={'std_states': 'polarization'})

    # Melt the dataframe to long format for easier plotting
    id_vars = ['t', 'type', 'scenario_grouped']
    value_vars = ['avg_state', 'polarization']
    data_long = pd.melt(data, id_vars=id_vars, value_vars=value_vars, 
                        var_name='measure', value_name='value')

    return data_long

def plot_network_dynamics(data, t_max=50, output_file=None):
    # Set style
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # Create the plot

    g = sns.relplot(
        data=data,
        x='t', y='value',
        hue='scenario_grouped',
        col='type',
        style='measure',  # Use 'measure' for line style
        kind='line',
        facet_kws={'sharex': False, 'sharey': False},
        height=5, aspect=1.2,
        legend='full'
    )

    # Customize the plot
    g.set_axis_labels("Time [timestep / system size]", "Value")
    g.set_titles("{col_name} Network")
    g.fig.suptitle("Time evolution of cooperativity and polarization for various network types", 
                   fontsize=16, fontweight='bold', y=1.02)

    # Adjust y-axis limits and add grid
    for ax in g.axes.flat:
        ax.set_ylim(-0.6, 1.1)
        for line in ax.lines:
            if 'polarization' in line.get_label():
                line.set_linestyle('--')

    # Adjust layout and legend
    g.tight_layout()
    g.fig.subplots_adjust(top=0.9, bottom=0.1)
    g.add_legend(title="Rewiring Scenarios", bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)

    # Save the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_single_topology_dynamics(data, t_max=1000, output_file=None):
    # Set style
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # Filter data
    data = data[data['t'] <= t_max]
    
    # Filter for specific network topologies
    filtered_data = data[
        ((data['type'] == 'sf') & (~data['scenario_grouped'].str.startswith('wtf'))) |
        ((data['type'] == 'DPAH') & (data['scenario_grouped'].str.startswith('wtf')))
    ]

    scenarios = filtered_data['scenario_grouped'].unique()
    scenarios = [s for s in scenarios if s != 'none_none']

    # Create a custom color palette
    n_colors = len(scenarios)
    custom_palette = sns.color_palette("husl", n_colors)
    color_dict = {scenario: color for scenario, color in zip(scenarios, custom_palette)}
    color_dict['none_none'] = 'gray'  # Consistent color for static network

    # Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    static_data = filtered_data[filtered_data['scenario_grouped'] == 'none_none']

    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        scenario_data = filtered_data[filtered_data['scenario_grouped'] == scenario]

        # Plot static network data
        for measure in ['avg_state', 'polarization']:
            static_measure = static_data[static_data['measure'] == measure]
            ax.plot(static_measure['t'], static_measure['value'], color='gray', 
                    linestyle='-' if measure == 'avg_state' else '--', alpha=0.5)

        # Plot scenario-specific data
        for measure in ['avg_state', 'polarization']:
            scenario_measure = scenario_data[scenario_data['measure'] == measure]
            ax.plot(scenario_measure['t'], scenario_measure['value'], color=color_dict[scenario],
                    linestyle='-' if measure == 'avg_state' else '--')

        ax.set_ylim(-0.6, 1.1)
        ax.set_title(f"{scenario.replace('_', ' ').capitalize()} ({scenario_data['type'].iloc[0]})")
        ax.set_xlabel("Time [timestep / system size]")
        ax.set_ylabel("Value")

    # Remove any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Time evolution of cooperativity and polarization for different scenarios", 
                 fontsize=16, fontweight='bold', y=1.02)

    # Create a custom legend
    legend_elements = [
        plt.Line2D([0], [0], color='gray', lw=2, label='Static network'),
        plt.Line2D([0], [0], color=next(iter(color_dict.values())), lw=2, label='Rewired network'),
        plt.Line2D([0], [0], color='black', lw=2, label='Cooperativity'),
        plt.Line2D([0], [0], color='black', linestyle='--', lw=2, label='Polarization')
    ]
    fig.legend(handles=legend_elements, title="Legend", bbox_to_anchor=(0.5, -0.05), 
               loc='upper center', ncol=4)

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.15)

    # Save the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()
#%%

# Main execution
if __name__ == "__main__":
    # Get list of relevant output files
    file_list = [f for f in os.listdir("../Output") if f.endswith(".csv") and "default_run_avg" in f]

    if not file_list:
        print("No suitable files found in the Output directory.")


    # Print file list with indices
    for i, file in enumerate(file_list):
        print(f"{i}: {file}")

    # Get user input for file selection
    file_index = int(input("Enter the index of the file you want to plot: "))

    if file_index < 0 or file_index >= len(file_list):
        print("Invalid file index.")


    # Load the data
    data = pd.read_csv(os.path.join("../Output", file_list[file_index]))

    # Set the maximum time step
    t_max = 70000  # You can adjust this value as needed

    get_N = file_list[file_index].split("_")[5]
    
    # Process the data
    processed_data = process_data(data, t_max)
    #assert 1 == 0
    # Generate plots
    today = date.today()
    plot_network_dynamics(processed_data, t_max, output_file=f"../Figs/Trajectories/network_dynamics_comparison_{get_N}_{today}.pdf")
    plot_single_topology_dynamics(processed_data, t_max, output_file=f"../Figs/Trajectories/single_topology_dynamics_comparison{get_N}_{today}.pdf")

    print("Plotting complete. Check the Figs directory for the output PDFs.")