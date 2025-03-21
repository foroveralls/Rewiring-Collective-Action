import sys
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from datetime import date

def setup_style():
    """Set up consistent styling across plots"""
    sns.set_theme()
    sns.set(style="ticks", font_scale=1.5)
    sns.set(rc={'axes.facecolor':'white', 
                'figure.facecolor':'white', 
                "axes.grid": True,
                "grid.color": 'black', 
                'grid.linestyle': 'dotted', 
                "axes.edgecolor": "black", 
                "patch.edgecolor": "black",
                "patch.linewidth": 0, 
                "axes.spines.bottom": True, 
                "grid.alpha": 0.5, 
                "xtick.bottom": True, 
                "ytick.left": True})

def find_inflection(seq):
    """Calculate inflection point in trajectory"""
    if len(seq) < 1200:  # Basic length check
        print(f"Warning: Sequence length {len(seq)} too short for reliable inflection detection")
        return False
    
    try:
        smooth = gaussian_filter1d(seq, 600)
        d2 = np.gradient(np.gradient(smooth))
        infls = np.where(np.diff(np.sign(d2)))[0]
        
        inf_min = 5000
        
        if not any(i >= inf_min for i in infls):
            return False
            
        for i in infls:
            if i < inf_min:
                continue
            else: 
                inf_ind = i
                break
        
        if inf_ind > inf_min and inf_ind < 20000:
            return inf_ind
        else:
            return False
            
    except Exception as e:
        print(f"Error in find_inflection: {str(e)}")
        return False

def estimate_convergence_rate(trajec, loc=None, regwin=10):
    """Estimate convergence rate around specified location"""
    x = np.arange(len(trajec) - 1)
    y = trajec 
    
    if loc is not None:
        x = x[loc-regwin: loc+regwin+1]
        y = trajec[loc-regwin: loc+regwin+1]
    
    y, x = np.asarray([y, x])
    idx = np.isfinite(x) & np.isfinite(y)
    
    # Linear regression
    n = np.size(x) 
    mx, my = np.mean(x), np.mean(y) 
    ssxy = np.sum(y*x) - n*my*mx 
    ssxx = np.sum(x*x) - n*mx*mx 
    b1 = ssxy / ssxx 
    b0 = my - b1*mx 
    
    rate = -b1/(trajec[loc]-1)
    return rate


def calculate_convergence_rates_with_sampling(data, sample_size=100):
    """
    Calculate convergence rates for sampled scenarios and model runs
    
    Parameters:
    -----------
    data : pandas DataFrame
        Input data containing all trajectories
    sample_size : int
        Number of trajectories to sample per scenario combination
    
    Returns:
    --------
    pandas DataFrame
        Convergence rates for sampled trajectories
    """
    rates_list = []
    
    # Group by scenario, rewiring, and type
    grouped = data.groupby(['scenario', 'rewiring', 'type'])
    
    print(f"\nProcessing samples from {len(grouped)} unique combinations")
    
    for name, group in grouped:
        scenario, rewiring, topology = name
        
        # Get unique model runs for this combination
        unique_runs = group['model_run'].unique()
        
        # Sample runs if we have more than sample_size
        if len(unique_runs) > sample_size:
            sampled_runs = np.random.choice(unique_runs, size=sample_size, replace=False)
            group = group[group['model_run'].isin(sampled_runs)]
        
        # Process each sampled run
        for run in group['model_run'].unique():
            trajectory = group[group['model_run'] == run]['avg_state'].values
            
            if len(trajectory) < 1200:
                continue
                
            inflection_x = find_inflection(trajectory)
            
            if inflection_x:
                try:
                    rate = estimate_convergence_rate(trajectory, loc=inflection_x)
                    rates_list.append({
                        'scenario': scenario,
                        'rewiring': rewiring,
                        'topology': topology,
                        'model_run': run,
                        'rate': rate * 1000  # Scale rate as in original code
                    })
                except Exception as e:
                    print(f"Error calculating rate for {scenario}_{rewiring}_{topology}_{run}: {str(e)}")
    
    return pd.DataFrame(rates_list)
def calculate_convergence_rates(data):
    """Calculate convergence rates for all scenarios and model runs"""
    rates_list = []
    
    # Group by scenario, rewiring, type, and model_run
    grouped = data.groupby(['scenario', 'rewiring', 'type', 'model_run'])
    
    print(f"\nProcessing {len(grouped)} unique combinations")
    
    for name, group in grouped:
        scenario, rewiring, topology, run = name
        
        # Get the state trajectory directly from avg_state column
        trajectory = group['avg_state'].values
        
        # print(f"\nAnalyzing: {scenario}_{rewiring}_{topology}_{run}")
        # print(f"Trajectory length: {len(trajectory)}")
        
        if len(trajectory) < 1200:
            print(f"Skipping - trajectory too short")
            continue
            
        inflection_x = find_inflection(trajectory)
        
        if inflection_x:
            try:
                rate = estimate_convergence_rate(trajectory, loc=inflection_x)
                rates_list.append({
                    'scenario': scenario,
                    'rewiring': rewiring,
                    'topology': topology,
                    'model_run': run,
                    'rate': rate * 1000  # Scale rate as in original code
                })
            except Exception as e:
                print(f"Error calculating rate: {str(e)}")
        else:
            print(f"No valid inflection point found")
            
    return pd.DataFrame(rates_list)
            
    return pd.DataFrame(rates_list)
def plot_violin(rates_df, output_path):
    """Create violin plot of convergence rates"""
    plt.figure(figsize=(14, 6))
    
    # Create combined scenario label
    rates_df['scenario_combined'] = rates_df['scenario'] + '_' + rates_df['rewiring']
    
    # Create violin plot without split
    sns.violinplot(
        data=rates_df,
        x='scenario_combined',
        y='rate',
        hue='topology',
        palette="Set2",
        inner='box',  # Shows quartiles inside violin
        scale='width'  # Makes all violins the same width
    )
    
    plt.title('Convergence Rate Distribution by Scenario', pad=20)
    plt.xlabel('Scenario', labelpad=10)
    plt.ylabel('Convergence Rate (×10³)', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Network Topology', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout() 
    plt.savefig(output_path.replace('.pdf', '_violin.pdf'), dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_violin_with_outlier_handling(rates_df, output_path, method="symlog"):
    """
    Create violin plot of convergence rates with improved outlier handling
    
    Parameters:
    -----------
    rates_df : pandas DataFrame
        DataFrame containing convergence rates and metadata
    output_path : str
        Path to save the output figure
    method : str
        Method for handling outliers:
        - "symlog": symmetric log transformation
        - "clip": clip outliers beyond percentiles
        - "split": create two subplots (main distribution and outliers)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Close any existing plots
    plt.close('all')
    
    def create_base_figure():
        rates_df['scenario_combined'] = rates_df['scenario'] + '_' + rates_df['rewiring']
        return rates_df['scenario_combined']
    
    if method == "symlog":
        fig = plt.figure(figsize=(14, 6))
        scenario_combined = create_base_figure()
        
        sns.violinplot(
            data=rates_df,
            x='scenario_combined',
            y='rate',
            hue='topology',
            palette="Set2",
            inner='box',
            scale='width'
        )
        
        plt.yscale('symlog', linthresh=np.std(rates_df['rate']))
        plt.title('Convergence Rates (Symmetric Log Scale)', pad=20)
        
    elif method == "clip":
        # Clip outliers beyond specified percentiles while indicating their presence
        lower_bound = np.percentile(rates_df['rate'], 5)
        upper_bound = np.percentile(rates_df['rate'], 95)
        
        rates_clipped = rates_df.copy()
        outlier_mask = (rates_df['rate'] < lower_bound) | (rates_df['rate'] > upper_bound)
        rates_clipped.loc[rates_clipped['rate'] < lower_bound, 'rate'] = lower_bound
        rates_clipped.loc[rates_clipped['rate'] > upper_bound, 'rate'] = upper_bound
        
        fig = plt.figure(figsize=(14, 6))
        scenario_combined = create_base_figure()
        
        sns.violinplot(
            data=rates_clipped,
            x='scenario_combined',
            y='rate',
            hue='topology',
            palette="Set2",
            inner='box',
            scale='width'
        )
        
        # Add markers for clipped regions
        plt.axhline(y=upper_bound, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=lower_bound, color='r', linestyle='--', alpha=0.5)
        plt.title('Convergence Rates (Clipped at 5th and 95th Percentiles)', pad=20)
        
    elif method == "split":
        # Create two subplots: main distribution and outliers
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
        scenario_combined = create_base_figure()
        
        # Main distribution (5-95 percentile)
        lower_bound = np.percentile(rates_df['rate'], 5)
        upper_bound = np.percentile(rates_df['rate'], 95)
        
        main_data = rates_df[rates_df['rate'].between(lower_bound, upper_bound)]
        outlier_data = rates_df[~rates_df['rate'].between(lower_bound, upper_bound)]
        
        # Main distribution plot
        sns.violinplot(
            data=main_data,
            x='scenario_combined',
            y='rate',
            hue='topology',
            palette="Set2",
            inner='box',
            scale='width',
            ax=ax1
        )
        ax1.set_title('Main Distribution (5-95 percentile)', pad=20)
        
        # Outliers plot
        sns.stripplot(
            data=outlier_data,
            x='scenario_combined',
            y='rate',
            hue='topology',
            palette="Set2",
            size=4,
            alpha=0.5,
            ax=ax2
        )
        ax2.set_title('Outliers', pad=10)
    
    # Common styling
    plt.xlabel('Scenario', labelpad=10)
    plt.ylabel('Convergence Rate (×10³)', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Network Topology', bbox_to_anchor=(1.05, 1))
    
    # Save figure
    plt.tight_layout()
    output_name = output_path.replace('.pdf', f'_violin_{method}.pdf')
    plt.savefig(output_name, dpi=600, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    # Close the figure to free memory
    plt.close(fig)
    
    # Common styling
    plt.xlabel('Scenario', labelpad=10)
    plt.ylabel('Convergence Rate (×10³)', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Network Topology', bbox_to_anchor=(1.05, 1))
    
    # Save figure
    plt.tight_layout()
    output_name = output_path.replace('.pdf', f'_violin_{method}.pdf')
    plt.savefig(output_name, dpi=600, bbox_inches='tight')
    plt.close()


def calculate_summary_statistics(rates_df, simplified=False):
    """
    Calculate summary statistics for convergence rates
    
    Parameters:
    -----------
    rates_df : pandas DataFrame
        DataFrame containing convergence rates and grouping variables
    simplified : bool
        If True, returns only mean and std. If False, includes full statistical analysis
    """
    from scipy import stats
    
    if simplified:
        # Simple summary with just means and standard deviations
        summary = rates_df.groupby(['scenario', 'rewiring', 'topology'])['rate'].agg([
            ('Mean Rate (×10³)', 'mean'),
            ('Std Dev (×10³)', 'std')
        ]).reset_index()
        
        # Round the numerical columns
        summary['Mean Rate (×10³)'] = summary['Mean Rate (×10³)'].round(3)
        summary['Std Dev (×10³)'] = summary['Std Dev (×10³)'].round(3)
        
        return summary
    
    # Detailed statistical summary
    results = []
    for (scenario, rewiring, topology), group in rates_df.groupby(['scenario', 'rewiring', 'topology']):
        rates = group['rate']
        n = len(rates)
        mean = rates.mean()
        std = rates.std(ddof=1)
        se = std / np.sqrt(n)
        
        # Calculate confidence interval
        confidence = 0.95
        t_value = stats.t.ppf((1 + confidence) / 2, df=n-1)
        margin_of_error = t_value * se
        ci_lower = mean - margin_of_error
        ci_upper = mean + margin_of_error
        
        results.append({
            'Scenario': scenario,
            'Rewiring': rewiring,
            'Topology': topology,
            'Mean Rate (×10³)': round(mean, 3),
            'SE (×10³)': round(se, 3),
            '95% CI': f"[{ci_lower:.3f}, {ci_upper:.3f}]",
            'n': n
        })
    
    return pd.DataFrame(results)

def save_summary_tables(rates_df, output_path):
    """
    Save both detailed and simplified summary statistics
    """
    # Get both types of summaries
    detailed_summary = calculate_summary_statistics(rates_df, simplified=False)
    simple_summary = calculate_summary_statistics(rates_df, simplified=True)
    
    # Save detailed summary
    detailed_summary.to_csv(f"{output_path}_detailed.csv", index=False)
    
    # Save simplified summary
    simple_summary.to_csv(f"{output_path}_simple.csv", index=False)
    
    # Create LaTeX tables
    detailed_latex = detailed_summary.style.to_latex(
        caption="Detailed Summary Statistics of Convergence Rates by Scenario",
        label="tab:convergence_rates_detailed",
        position="htbp",
        column_format="lcccccr",
        hrules=True
    )
    
    simple_latex = simple_summary.style.to_latex(
        caption="Summary of Convergence Rates by Scenario",
        label="tab:convergence_rates_simple",
        position="htbp",
        column_format="lcccc",
        hrules=True
    )
    
    # Save LaTeX tables
    with open(f"{output_path}_detailed.tex", 'w') as f:
        f.write(detailed_latex)
    
    with open(f"{output_path}_simple.tex", 'w') as f:
        f.write(simple_latex)
    
    return detailed_summary, simple_summary


def main(saved_rates = None):
    
    if saved_rates is None:
        # Load the data
        file_list = [f for f in os.listdir("../Output") if "default_run_individual" in f and f.endswith(".csv")]
        
        if not file_list:
            raise FileNotFoundError("No individual run files found in ../Output directory")
        
        # Print file list with indices
        for i, file in enumerate(file_list):
            print(f"{i}: {file}")
        
        # Get user input for file selection
        file_index = int(input("Enter the index of the file you want to plot: "))
        
        # Load the selected file
        data_path = os.path.join("../Output", file_list[file_index])
        
        # # Load the most recent file
        # latest_file = max(file_list)
        # data_path = os.path.join("../Output", latest_file)
        print(f"Processing file: {file_list[file_index]}")
        
        # Read the data
        raw_data = pd.read_csv(data_path, low_memory=False)
    
        print(f"\nData shape: {raw_data.shape}")
        print(f"Columns: {raw_data.columns.tolist()}")
        
        # Fill NA values in categorical columns
        raw_data['rewiring'] = raw_data['rewiring'].fillna('none')
        raw_data['scenario'] = raw_data['scenario'].fillna('none')
        
        # No need to melt the data - we can use it directly
        print("\nSample of data:")
        print(raw_data.head())
        
        # Calculate convergence rates
        #rates_df = calculate_convergence_rates(raw_data)
        
        # Calculate convergence rates with sampling
        rates_df = calculate_convergence_rates_with_sampling(raw_data, sample_size=100)
        
    
    else:
        rates_df = saved_rates
        

    
    if len(rates_df) == 0:
        print("Warning: No valid convergence rates calculated")
        return
    
    # Print summary before plotting
    # print("\nUnique scenarios:", rates_df['scenario'].unique())
    # print("Unique topologies:", rates_df['topology'].unique())
    # print("Unique rewiring:", rates_df['rewiring'].unique())
    
    # Setup output directory
    os.makedirs("../Figs/Convergence", exist_ok=True)
    N = file_list[file_index].split("_")[5]  # Extract N from filename
    base_output_path = f"../Figs/Convergence/convergence_rates_N{N}_{date.today()}.pdf"
    
    # Set up styling
    setup_style()
    
    # Generate all plot types
    #plot_violin(rates_df, base_output_path)
    
    # Generate plots with different outlier handling methods
    plot_violin_with_outlier_handling(rates_df, base_output_path, method="symlog")
    plot_violin_with_outlier_handling(rates_df, base_output_path, method="clip")
    #plot_violin_with_outlier_handling(rates_df, base_output_path, method="split")
   
    
    # Calculate and save summary statistics
    summary_path = f"../Output/convergence_summary_N{N}_{date.today()}"
    detailed_stats, simple_stats = save_summary_tables(rates_df, summary_path)
       
    # Print both summaries to console
    print("\nSimplified Summary Statistics:")
    print(simple_stats.to_string(index=False))
    print("\nDetailed Summary Statistics:")
    print(detailed_stats.to_string(index=False))
     
    return simple_stats, rates_df

if __name__ == "__main__":
    stats, rate = main()