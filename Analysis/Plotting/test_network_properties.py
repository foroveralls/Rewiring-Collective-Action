#!/usr/bin/env python3
"""
Test script for network properties plotting
"""

import os
import sys
import glob

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_snapshot_files(data_dir="../../Output"):
    """Check what snapshot files are available"""
    print(f"Checking for snapshot files in {data_dir}")
    
    # Look for various potential snapshot file patterns
    patterns = [
        "*_snapshots.pkl",
        "*snapshots*.pkl", 
        "*_model_*.pkl"
    ]
    
    found_files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(data_dir, pattern))
        found_files.extend(files)
    
    if found_files:
        print(f"Found {len(found_files)} potential snapshot files:")
        for f in found_files:
            print(f"  - {os.path.basename(f)}")
    else:
        print("No snapshot files found")
        print("Available files in Output directory:")
        all_files = glob.glob(os.path.join(data_dir, "*"))
        for f in all_files[:10]:  # Show first 10 files
            print(f"  - {os.path.basename(f)}")
        if len(all_files) > 10:
            print(f"  ... and {len(all_files) - 10} more files")
    
    return found_files

def test_basic_imports():
    """Test basic imports without external dependencies"""
    print("Testing basic imports...")
    
    try:
        import pandas as pd
        print("✓ pandas available")
    except ImportError:
        print("✗ pandas not available")
    
    try:
        import numpy as np
        print("✓ numpy available")
    except ImportError:
        print("✗ numpy not available")
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib available")
    except ImportError:
        print("✗ matplotlib not available")
    
    try:
        import networkx as nx
        print("✓ networkx available")
    except ImportError:
        print("✗ networkx not available")
    
    try:
        import pickle
        print("✓ pickle available")
    except ImportError:
        print("✗ pickle not available")

if __name__ == "__main__":
    print("=== Testing Network Properties Plot Setup ===")
    test_basic_imports()
    print("\n=== Checking for snapshot data ===")
    snapshot_files = check_snapshot_files()
    
    if snapshot_files:
        print(f"\n✓ Ready to process {len(snapshot_files)} snapshot files")
    else:
        print("\n! No snapshot files found - you'll need to run simulations with save_snapshots=True first")