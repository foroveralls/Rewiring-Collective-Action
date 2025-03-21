import sys
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import FuncFormatter
from datetime import date

# Set the seaborn theme parameters


cm = 1/2.54
# Global settings
FONT_SIZE = 14
FRIENDLY_COLORS = {
    'static': '#EE7733',      # Orange
    'random': '#0077BB',      # Blue
    'local (similar)': '#33BBEE',    # Cyan
    'local (opposite)': '#009988',   # Teal
    'bridge (similar)': '#CC3311',   # Red
    'bridge (opposite)': '#EE3377',  # Magenta
    'wtf': '#BBBBBB',         # Grey
    'node2vec': '#44BB99'     # Blue-green
}




def setup_plotting_style():
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
        'figure.figsize': (11.4*cm, 7*cm), 
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out'
    })
    
    sns.set_theme(font_scale=FONT_SIZE/12)
    sns.set(style="ticks")
    sns.set(rc={
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        "axes.grid": True,
        "grid.color": 'black',  # Changed to white for heatmap
        'grid.linestyle': 'solid', 
        "axes.edgecolor": "black",
        "patch.edgecolor": "black",
        "patch.linewidth": 0.5,  # Reduced from original
        "axes.spines.bottom": True,
        "grid.alpha": 0.4,  # Increased for better visibility
        "xtick.bottom": True,
        "ytick.left": True
    })
    
    pass
# Set the seaborn theme parameters

sns.set_theme(font_scale=FONT_SIZE/14)  # Default font size is 12, so scale accordingly
sns.set(style="ticks")
sns.set(rc={'axes.facecolor':'white', 
            'font.family': 'Arial',
            'font.size': 14,
            'figure.facecolor':'white', 
            "axes.grid": True,
            "grid.color": 'black', 
            'grid.linestyle': 'dotted', 
            "axes.edgecolor": "black", 
            "patch.edgecolor": "black",
            "patch.linewidth": 0.5, 
            "axes.spines.bottom": True, 
            "grid.alpha": 0.8, 
            "xtick.bottom": True, 
            "ytick.left": True})


# Define friendly names mapping
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

def find_inflection(seq, min_idx=5000, sigma=300):
    """Calculate inflection point in trajectory"""
    smooth = gaussian_filter1d(seq, sigma)
    d2 = np.gradient(np.gradient(smooth))
    infls = np.where(np.diff(np.sign(d2)))[0]
    
    # Find first inflection point after min_idx
    inf_ind = None
    for i in infls:
        if i >= min_idx:
            inf_ind = i
            break
    
    # If no inflection after min_idx but we have some inflection points, take the last one
    if inf_ind is None and len(infls) > 0:
        inf_ind = infls[-1]
    
    # No inflection points found
    if inf_ind is None:
        return False, smooth
    
    # Ensure inflection point is within reasonable range
    if inf_ind < min_idx or inf_ind >= len(seq) * 0.9:  # Avoid points too close to the end
        return False, smooth
    
    return inf_ind, smooth
def estimate_convergence_rate(trajec, loc=None, regwin=15):
    """Estimate convergence rate around specified location"""
    if loc is None or not isinstance(loc, (int, np.integer)):
        return 0
    
    x = np.arange(len(trajec))
    y = trajec
    
    if loc is not None:
        # Ensure regression window doesn't go out of bounds
        start_idx = max(0, loc-regwin)
        end_idx = min(len(trajec)-1, loc+regwin+1)
        x = x[start_idx:end_idx]
        y = trajec[start_idx:end_idx]
    
    # Ensure we have enough points for regression
    if len(x) < 3:
        return 0
    
    # Linear regression
    n = np.size(x) 
    mx, my = np.mean(x), np.mean(y) 
    ssxy = np.sum(y*x) - n*my*mx 
    ssxx = np.sum(x*x) - n*mx*mx 
    
    # Avoid division by zero
    if ssxx == 0:
        return 0
        
    b1 = ssxy / ssxx 
    b0 = my - b1*mx 
    
    # Protect against division by zero or values close to 1
    if abs(trajec[loc] - 1) < 0.001:
        rate = -b1/0.001
    else:
        rate = -b1/(trajec[loc]-1)
    
    return rate


def calculate_convergence_rates(data):
    """Calculate convergence rates for all scenarios"""
    rates_list = []
    
    # Set parameters
    SIGMA = 300
    MIN_IDX = 5000
    REGWIN = 15
    
    # Group by scenario and type
    grouped = data.groupby(['scenario_grouped', 'type'])
    
    for name, group in grouped:
        scenario, topology = name
        
        # Get trajectory for this scenario
        trajectory = group[group['measurement'] == 'avg_state']['value'].values
        
        # Find inflection point and get smoothed trajectory
        inflection_x, smoothed = find_inflection(trajectory, min_idx=MIN_IDX, sigma=SIGMA)
        
        # Calculate rate
        if inflection_x:
            rate = estimate_convergence_rate(smoothed, loc=inflection_x, regwin=REGWIN)
        else:
            rate = 0
            
        # Map to friendly name
        friendly_scenario = friendly_names.get(scenario, scenario)
            
        rates_list.append({
            'scenario': friendly_scenario,
            'topology': topology,
            'rate': rate * 1000  # Scale rate for display
        })
    
    return pd.DataFrame(rates_list)



def plot_convergence_rates(rates_df, output_path):
    """Create convergence rate comparison plot with scenarios ordered by median convergence rate,
    using traditional statistical physics markers in black and white"""
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.figure(figsize=(11, 8))
    
    # Calculate median rate for each scenario to determine order
    scenario_medians = rates_df.groupby('scenario')['rate'].median().sort_values(ascending=False)
    scenario_order = scenario_medians.index.tolist()
    
    # Create a new categorical column with ordered scenarios
    rates_df = rates_df.copy()  # Create a copy to avoid SettingWithCopyWarning
    rates_df['scenario'] = pd.Categorical(rates_df['scenario'], 
                                        categories=scenario_order, 
                                        ordered=True)
    
    # Define traditional statistical physics markers and their labels
    topology_markers = {
        'DPAH': 'x',      # cross
        'cl': '+',        # plus
        'Twitter': '*',   # asterisk
        'FB': '.'         # point
    }
    
    # Create x-coordinates for scenarios
    x_coords = np.arange(len(scenario_order))
    
    # Create main scatter plot with topology markers
    for topology, marker in topology_markers.items():
        mask = rates_df['topology'] == topology
        scenario_data = rates_df[mask]
        
        # Map scenarios to x-coordinates
        scenario_x = [x_coords[list(scenario_order).index(s)] for s in scenario_data['scenario']]
        
        plt.scatter(
            scenario_x,
            scenario_data['rate'],
            marker=marker,
            s=100,  # Marker size
            c='black',  # All markers in black
            label=topology,
            alpha=0.7
        )
    
    # Add median values as horizontal lines
    for i, scenario in enumerate(scenario_order):
        median_rate = scenario_medians[scenario]
        plt.hlines(y=median_rate, xmin=i-0.2, xmax=i+0.2, 
                  colors='red', linestyles='solid', alpha=0.5)
    
    # Customize the plot
    plt.title('Convergence Rates Comparison', pad=20, fontsize=FONT_SIZE)
    plt.xlabel('Rewiring Strategy', labelpad=10, fontsize=FONT_SIZE)
    plt.ylabel('Convergence Rate (×10³)', labelpad=10, fontsize=FONT_SIZE)
    
    # Set x-axis ticks and labels
    plt.xticks(x_coords, scenario_order, 
               rotation=45, 
               horizontalalignment='right',
               rotation_mode='anchor',
               fontsize=FONT_SIZE)
    
    # Create legend for topologies (markers)
    legend_elements = [plt.Line2D([0], [0], marker=marker, color='black', 
                                label=topology, markersize=10, linestyle='None')
                      for topology, marker in topology_markers.items()]
    
    # Add median line to legend
    legend_elements.append(plt.Line2D([0], [0], color='red', alpha=0.5,
                                    label='Median', linestyle='solid'))
    
    # Place legend
    plt.legend(handles=legend_elements, title="Network Topology", 
              loc='upper left', bbox_to_anchor=(0.80, 0.95))
    
    plt.yticks(fontsize=FONT_SIZE)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout to accommodate legend
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, 
                dpi=300, 
                bbox_inches='tight', 
                format='pdf',
                metadata={'Creator': "Jordan", 'Producer': None})
    plt.show()
    plt.close()

def main():
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
    
    # Load the selected file
    data_path = os.path.join("../Output", file_list[file_index])
    raw_data = pd.read_csv(data_path)
    
    # Prepare data for analysis
    id_vars = ['t', 'scenario', 'rewiring', 'type']
    raw_data['rewiring'] = raw_data['rewiring'].fillna('none')
    raw_data['scenario'] = raw_data['scenario'].fillna('none')
    
    # Melt the dataframe
    melted_data = pd.melt(raw_data, id_vars=id_vars, 
                         var_name='measurement', value_name='value')
    melted_data['scenario_grouped'] = melted_data['scenario'].str.cat(
        melted_data['rewiring'], sep='_')
    
    # Calculate convergence rates
    rates_df = calculate_convergence_rates(melted_data)
    
    # Create output directory if it doesn't exist
    os.makedirs("../Figs/Convergence", exist_ok=True)
    
    # Generate output filename
    N = file_list[file_index].split("_")[5]  # Extract N from filename
    output_path = f"../Figs/Convergence/convergence_rates_N{N}_{date.today()}.pdf"
    
    setup_plotting_style()
    
    # Create and save the plot
    plot_convergence_rates(rates_df, output_path)
    print(f"Convergence rate plot saved to: {output_path}")
    
    # Print summary statistics with friendly names
    print("\nConvergence Rate Summary Statistics:")
    (rates_df.groupby(['scenario'])['rate'].describe())
    print()
    
    return rates_df

if __name__ == "__main__":
   df_rates = main()