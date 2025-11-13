"""
Compute predictions and learned features for the dataset generated in dataset.py.
Uses logic from genome_stats.py.
"""

import os
import sys
import numpy as np
import pandas as pd

# our imports
from dataset import load_data
from utils import BATCH_SIZE, apply_seed_to_path, compute_all_for_dataset, get_model, iterate_seeds, preds_lf_path, save_preds_lf

def analyze_timing_statistics(timing_data):
    """
    Compute mean and standard deviation for the pred/lf computation time.
    
    Parameters:
    -----------
    timing_data : list
        List containing timing information for the computation
    """
    
    times_array = np.array(timing_data)
    summary = {
        'mean_time': np.mean(times_array),
        'std_time': np.std(times_array),
        'min_time': np.min(times_array),
        'max_time': np.max(times_array),
        'total_time': np.sum(times_array),
        'n_samples': len(times_array),
        'batch_size': BATCH_SIZE
    }
    
    print(summary)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("./figdata/timing_analysis_preds.csv", index=False)


def main(pop, model_path, fc_size, max_samples=None, benchmark=True):
    # Load the dataset
    print("Loading dataset...")
    samples, _ = load_data(pop)
    print(f"Dataset shape: {samples.shape}")
    
    if max_samples is not None:
        print(f"Processing only {max_samples} samples")

    if benchmark:
        all_timings = []

    for seed in iterate_seeds(model_path, stop=5 if "random" in model_path else 20):
        model_name = os.path.basename(apply_seed_to_path(model_path, seed)).split(".")[0]
        print(f"\nProcessing model {model_name}...")
        model = get_model(model_path, samples[0:1], seed=seed, 
                          fc_size=fc_size,
                          add_norm=("discs" in model_path))

        if os.path.exists(preds_lf_path(pop, model_name)):
            print(f"Results for model {model_name} already exist. Skipping...")
            continue

        # Compute preds and lf
        if benchmark:
            preds, lf, timing_data = compute_all_for_dataset(model, samples, max_samples=max_samples, benchmark=True)
            all_timings.extend(timing_data)
            
        else:
            preds, lf = compute_all_for_dataset(model, samples, max_samples=max_samples, benchmark=False)
        
        # Save results
        save_preds_lf(pop, model_name, preds, lf)

    # Print shape of everything
    # print(f"Predictions shape: {preds.shape}")
    # print(f"Learned features shape: {lf.shape}")

    # Analyze timing
    if benchmark:
        analyze_timing_statistics(all_timings)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nComputes preds and lf for all seeds (filtered) like <model_path>\n")   
        print("Usage: python compute_preds.py <model_path> [fc_size]")
        print("Example: python compute_preds.py discs/disc_N.keras 64")
        sys.exit(1)

    model_path = sys.argv[1]
    fc_size = 64 if len(sys.argv) < 3 else sys.argv[2]

    main(pop, model_path, fc_size, max_samples=None, benchmark=True)