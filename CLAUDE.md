# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for studying network rewiring effects on polarization and collective action using agent-based models. The project analyzes how different rewiring strategies affect cooperative consensus formation and depolarization in social networks.It will be published in a high impact journal like PNAS or Nature Communications so figures etc have to be high quality and follow good design standards appropriate for the journals. Respond like an expert in network science, statisical physics, and computational social science. Please be conservative with code changes and try and integrate suggestions with existing architecture.

## Environment Setup

The project uses conda for environment management:

```bash
# Create environment from environment.yaml
conda env create

# Activate environment
conda activate collective_rewiring
```

## Key Architecture

### Core Model Components
- `Analysis/models_checks.py`: Contains the main agent-based model implementation
- `Analysis/run.py`: Main execution script for running simulations
- `Analysis/sweep_utils.py`: Utilities for parameter sweeps and configuration management

### Analysis Structure
- `Analysis/`: Main analysis code and model implementations
- `Analysis/Plotting/`: Visualization scripts for generating figures
- `Analysis/Stats/`: Statistical analysis scripts
- `Auxillary/`: Supporting utilities including network stats and fast implementations
- `Pre_processing/`: Network preprocessing and empirical data analysis
- `Output/`: Generated results and processed data
- `Figs/`: Generated figures and visualizations

### Performance Optimization
- `Auxillary/fast_wtf/`: Rust implementation for performance-critical computations
  - Uses PyO3 for Python bindings
  - Build with: `cargo build --release` (from the fast_wtf directory)
- Multiprocessing support with optimal CPU allocation (reserves 25% of cores for system)

### Parameter Sweeps
Scripts ending in `*parameter_sweep*` or `*_sweep.py` handle sensitivity analyses:
- Use multiprocessing for parallel execution
- Generate sweep IDs and save configurations automatically
- Results saved to Output/ directory

### Network Processing
- Supports multiple network formats (NetworkX, igraph, rustworkx)
- Empirical networks stored in `Pre_processing/networks_processed/`
- Uses node2vec for network embeddings when needed

## Common Development Commands

```bash
# Run main simulation
cd Analysis && python run.py

# Run parameter sweeps
cd Analysis && python general_param_sweep.py

# Build Rust performance components
cd Auxillary/fast_wtf && cargo build --release

# Generate plots
cd Analysis/Plotting && python <specific_plot_script>.py
```

## Key Dependencies
- NetworkX, igraph, rustworkx for network analysis
- netin for network generation and analysis
- NumPy, SciPy, pandas for data processing
- Matplotlib, seaborn for visualization
- multiprocessing, joblib for parallel processing
- PyO3 for Rust-Python integration