# sweep_utils.py

import time
import random
import string
import os
from datetime import date

def get_sweep_id(parameter_names):
    """Generate a unique identifier for the entire parameter sweep."""
    timestamp = time.strftime("%Y%m%d_%H%M")
    param_str = "_".join(parameter_names)
    random_str = ''.join(random.choices(string.ascii_lowercase, k=3))
    return f"sweep_{timestamp}_{param_str}_{random_str}"

def save_sweep_config(sweep_id, parameter_names, parameters, combined_list, 
                     num_simulations, base_args, output_dir="../Output"):
    """Save the parameter sweep configuration to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    config_file = os.path.join(output_dir, f"sweep_config_{sweep_id}.txt")
    
    total_params = len(list(parameters.values())[0]) if parameters else 0
    total_sims = total_params * len(combined_list) * num_simulations
    
    with open(config_file, "w") as f:
        f.write(f"Sweep ID: {sweep_id}\n")
        f.write(f"Date: {date.today()}\n")
        f.write(f"Sims per combo: {num_simulations}\n")
        f.write(f"Total sims: {total_sims}\n\n")
        
        f.write("Swept parameters:\n")
        for param_name, param_values in parameters.items():
            if hasattr(param_values, '__iter__') and not isinstance(param_values, str):
                f.write(f"  {param_name}: {len(param_values)} values [{min(param_values):.4f}, {max(param_values):.4f}]\n")
            else:
                f.write(f"  {param_name}: {param_values}\n")
        
        f.write(f"\nScenarios ({len(combined_list)}):\n")
        for algo, mode, topology in combined_list:
            f.write(f"  {algo}_{mode}_{topology}\n")
        
        f.write("\nFixed params:\n")
        excluded = set(parameter_names) | {"rewiringAlgorithm", "rewiringMode", "type", "top_file", "nwsize"}
        for key, value in sorted(base_args.items()):
            if key not in excluded:
                f.write(f"  {key}: {value}\n")
    
    print(f"Config saved: {config_file}")