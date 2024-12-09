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

def plot_barplot(rates_df, output_path):
    """Create bar plot with error bars"""
    plt.figure(figsize=(14, 6))
    
    # Create combined scenario label
    rates_df['scenario_combined'] = rates_df['scenario'] + '_' + rates_df['rewiring']
    
    # Calculate mean and standard error correctly
    summary = rates_df.groupby(['scenario_combined', 'topology'])['rate'].agg([
        'mean',
        'count',
        'std'
    ]).reset_index()
    
    # Calculate standard error
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    
    # Create grouped bar plot using the summary data
    bar_plot = sns.barplot(
        data=summary,
        x='scenario_combined',
        y='mean',
        hue='topology',
        palette="Set2",
        alpha=0.8,
        capsize=0.05,
        errwidth=1,
        errorbar=('ci', 68)  # This shows 1 standard error
    )
    
    plt.title('Mean Convergence Rates by Scenario', pad=20)
    plt.xlabel('Scenario', labelpad=10)
    plt.ylabel('Mean Convergence Rate (×10³)', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Network Topology', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.pdf', '_bar.pdf'), dpi=600, bbox_inches='tight')
    plt.show()
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


def main():
    # Load the data
    file_list = [f for f in os.listdir("../Output") if "default_run_individual" in f and f.endswith(".csv")]
    
    if not file_list:
        raise FileNotFoundError("No individual run files found in ../Output directory")
    
    # Load the most recent file
    latest_file = max(file_list)
    data_path = os.path.join("../Output", latest_file)
    print(f"Processing file: {latest_file}")
    
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
    rates_df = calculate_convergence_rates(raw_data)
    
    if len(rates_df) == 0:
        print("Warning: No valid convergence rates calculated")
        return
    
    # Print summary before plotting
    # print("\nUnique scenarios:", rates_df['scenario'].unique())
    # print("Unique topologies:", rates_df['topology'].unique())
    # print("Unique rewiring:", rates_df['rewiring'].unique())
    
    # Setup output directory
    os.makedirs("../Figs/Convergence", exist_ok=True)
    N = latest_file.split("_")[5]  # Extract N from filename
    base_output_path = f"../Figs/Convergence/convergence_rates_N{N}_{date.today()}.pdf"
    
    # Set up styling
    setup_style()
    
    # Generate all plot types
    plot_violin(rates_df, base_output_path)
    
    # Calculate and save summary statistics
    summary_path = f"../Output/convergence_summary_N{N}_{date.today()}"
    detailed_stats, simple_stats = save_summary_tables(rates_df, summary_path)
       
    # Print both summaries to console
    print("\nSimplified Summary Statistics:")
    print(simple_stats.to_string(index=False))
    print("\nDetailed Summary Statistics:")
    print(detailed_stats.to_string(index=False))
     
    return simple_stats

if __name__ == "__main__":
    rate = main()