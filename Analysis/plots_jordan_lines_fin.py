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

def plot_single_topology_dynamics(data, t_max=50, output_file=None):
    # Set style
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # Filter data for single topology per scenario
    filtered_data = data[
        ((data['type'] == 'cl') & (data['scenario'] != 'wtf')) |
        ((data['type'] == 'DPAH') & (data['scenario'] == 'wtf'))
    ]

    if filtered_data.empty:
        print("No data to plot after filtering.")
        return
    
    g = sns.relplot(
      data=filtered_data,
      x='t', y='value',
      hue='rewiring',
      col='scenario',
      style='measure',  # Use 'measure' for line style
      kind='line',
      facet_kws={'sharex': False, 'sharey': False},
      height=5, aspect=1.2,
      legend='full'
      )

    # Customize the plot
    g.set_axis_labels("Time [timestep / system size]", "Value")
    g.set_titles("{col_name}")
    g.fig.suptitle("Time evolution of cooperativity and polarization for different scenarios", 
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
    g.add_legend(title="Rewiring Modes", bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)

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
    t_max = 1000  # You can adjust this value as needed

    # Process the data
    processed_data = process_data(data, t_max)
    #assert 1 == 0
    # Generate plots
    today = date.today()
    plot_network_dynamics(processed_data, t_max, output_file=f"../Figs/network_dynamics_comparison_{today}.pdf")
    plot_single_topology_dynamics(processed_data, t_max, output_file=f"../Figs/single_topology_dynamics_comparison_{today}.pdf")

    print("Plotting complete. Check the Figs directory for the output PDFs.")