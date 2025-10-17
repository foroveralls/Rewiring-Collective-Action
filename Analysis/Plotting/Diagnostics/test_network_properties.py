#!/usr/bin/env python3
"""
Test script for network properties plotting
"""

import os
import sys
import glob
import gzip
import pickle

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_snapshot_files(data_dir="../../Output"):
    """Check what snapshot files are available"""
    print(f"Checking for snapshot files in {data_dir}")
    
    # Look for various potential snapshot file patterns (including gz files)
    patterns = [
        "*_snapshots.pkl.gz",
        "*snapshots*.pkl.gz",
        "*_model_*.pkl.gz",
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

if __name__ == "__main__":
    print("=== Testing Network Properties Plot Setup ===")
    print("\n=== Checking for snapshot data ===")
    snapshot_files = check_snapshot_files()
    
    if snapshot_files:
        print(f"\n✓ Ready to process {len(snapshot_files)} snapshot files")
    else:
        print("\n! No snapshot files found - you'll need to run simulations with save_snapshots=True first")