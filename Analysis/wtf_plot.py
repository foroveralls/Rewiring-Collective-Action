#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
df = pd.read_csv("out.csv")

# Get unique algorithms
algorithms = sorted(df['algorithm'].unique())
n_algos = len(algorithms)

# Create subplot grid
cols = 3
rows = (n_algos + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
axes = axes.flatten() if n_algos > 1 else [axes]

# Plot each algorithm in separate subplot
for i, algo in enumerate(algorithms):
    ax = axes[i]
    algo_data = df[df['algorithm'] == algo]
    runs = sorted(algo_data['run'].unique())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(runs)))
    
    for j, run in enumerate(runs):
        run_data = algo_data[algo_data['run'] == run].sort_values('time')
        ax.plot(run_data['time'], run_data['state'], 
               color=colors[j], linewidth=1.5, alpha=0.8, label=f'Run {run}')
    
    ax.set_title(algo)
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.grid(True, alpha=0.3)
    ax.legend()

# Hide empty subplots
for i in range(n_algos, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# Print summary
print("Data summary:")
for algo in algorithms:
    algo_data = df[df['algorithm'] == algo]
    runs = len(algo_data['run'].unique())
    points = len(algo_data)
    print(f"{algo}: {runs} runs, {points} total points")