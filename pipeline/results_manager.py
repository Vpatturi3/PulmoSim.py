#!/usr/bin/env python3
"""
Results Manager - Auto-numbering system for results folders
Creates folders like results1, results2, results3, etc.
"""

import os
import glob
import re


def get_next_results_folder(base_dir="."):
    """
    Find the next available results folder number.
    Returns the path like 'results/results1', 'results/results2', etc.
    """
    # Ensure main results directory exists
    results_main_dir = os.path.join(base_dir, "results")
    os.makedirs(results_main_dir, exist_ok=True)
    
    # Look for existing results folders with numbers inside results/
    results_pattern = os.path.join(results_main_dir, "results*")
    existing_folders = glob.glob(results_pattern)
    
    # Extract numbers from existing folders
    numbers = []
    for folder in existing_folders:
        folder_name = os.path.basename(folder)
        # Match 'results' followed by digits
        match = re.match(r'results(\d+)$', folder_name)
        if match:
            numbers.append(int(match.group(1)))
    
    # Find next available number
    if not numbers:
        next_num = 1
    else:
        next_num = max(numbers) + 1
    
    results_folder = f"results{next_num}"
    results_path = os.path.join(results_main_dir, results_folder)
    
    # Create the folder
    os.makedirs(results_path, exist_ok=True)
    
    return results_path


def create_results_folder(base_dir="."):
    """
    Create and return the next results folder path inside results/
    """
    return get_next_results_folder(base_dir)


if __name__ == "__main__":
    # Test the function
    folder = create_results_folder()
    print(f"Created results folder: {folder}")