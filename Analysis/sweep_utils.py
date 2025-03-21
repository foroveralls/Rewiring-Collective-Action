# sweep_utils.py

import time
import random
import string
from datetime import date

def get_sweep_id(parameter_names):
    """Generate a unique identifier for the entire parameter sweep.
    
    Args:
        parameter_names: List of parameter names being swept
        
    Returns:
        A unique sweep identifier string
    """

    # Create timestamp component
    timestamp = time.strftime("%Y%m%d_%H%M")
    
    # Create parameter component (what's being swept)
    param_str = "_".join(parameter_names)
    
    # Add random component (3 characters)
    random_str = ''.join(random.choices(string.ascii_lowercase, k=3))
    
    # Combine components
    sweep_id = f"sweep_{timestamp}_{param_str}_{random_str}"
    
    return sweep_id

def save_sweep_config(sweep_id, parameter_names, parameters, combined_list, 
                     num_simulations, models_checks_module, output_dir="../Output"):
    """Save the parameter sweep configuration to a text file.
    
    Args:
        sweep_id: Unique identifier for this sweep
        parameter_names: List of parameter names being swept
        parameters: Dictionary of parameters and their values
        combined_list: List of (algorithm, mode, topology) combinations
        num_simulations: Number of simulations per parameter combination
        models_checks_module: The imported models_checks module
        output_dir: Directory to save the configuration file
    """
    # Ensure output directory exists
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the output file path
    config_file = os.path.join(output_dir, f"sweep_config_{sweep_id}.txt")
    
    with open(config_file, "w") as f:
        f.write(f"Parameter Sweep ID: {sweep_id}\n")
        f.write(f"Date: {date.today()}\n")
        f.write(f"Number of simulations per parameter combination: {num_simulations}\n\n")
        
        f.write("Parameters being swept:\n")
        for param_name, param_values in parameters.items():
            f.write(f"  {param_name}: {param_values}\n")
        
        f.write("\nScenarios being tested:\n")
        for algo, mode, topology in combined_list:
            f.write(f"  {algo}_{mode}_{topology}\n")
        
        f.write("\nFixed parameters:\n")
        fixed_params = models_checks_module.getargs()
        for key, value in sorted(fixed_params.items()):
            if key not in parameters and key not in ["rewiringAlgorithm", "rewiringMode", "type", "top_file"]:
                f.write(f"  {key}: {value}\n")
    
    print(f"Sweep configuration saved to: {config_file}")