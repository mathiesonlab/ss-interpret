"""
Compute summary statistics for the dataset generated in dataset.py.
Uses logic from genome_stats.py to compute stats for both real and simulated data.
SM: modifying for mosquito data
"""

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# our imports
from pg_gan import ss_helpers, ss_extra
from dataset import load_data, metadata_file
from utils import POP1_n, POP2_n

def split_matrices(matrices, sample_sizes):

    # set up empty array
    _all = []

    start_idx = 0
    for s in sample_sizes:
        end_idx = start_idx + s

        # parse matrices
        p = matrices[:,start_idx:end_idx,:,:]
        _all.append(p)

        # last step: update start_idx
        start_idx = end_idx

    return _all

# Compute original summary stats
def flatten(data):
    for item in data:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield float(item)

def compute_stats_for_dataset(samples, max_samples=None, benchmark=True):
    """
    Compute summary statistics for the entire dataset.
    
    Parameters:
    -----------
    samples : np.ndarray
        Array of shape (n_samples, num_snps, 2) containing the genomic data
    max_samples : int, optional
        Maximum number of samples to process. If None, process all samples.
    benchmark : bool
        Whether to collect timing information for each statistic
        
    Returns:
    --------
    all_stats : np.ndarray
        Array of shape (n_samples, num_stats) containing computed statistics
    timing_data : dict
        Dictionary containing timing information for each statistic (if benchmark=True)
    """
    
    if max_samples is not None:
        samples = samples[:max_samples]
    
    print(f"Computing summary statistics for {len(samples)} samples...")
    if benchmark:
        print("Benchmarking enabled - collecting timing data...")

    # split into populations if more than one
    sample_sizes = [POP1_n, POP2_n]
    samples_per_pop = split_matrices(samples, sample_sizes)
    
    # list of stats per pop
    aggregated_stats = []
    for pop in samples_per_pop:
        all_stats, timing_data = stats_per_pop(pop)
        aggregated_stats.append(all_stats)
    aggregated_stats = np.array(aggregated_stats)

    # compute Fst
    fst, timing = ss_helpers.fst_all(samples, sample_sizes, benchmark=True)

    # reshape into n_samples x n_stats
    n_samples = len(samples)
    final_stats = []
    for i in range(n_samples):
        final_stats.append(np.concatenate((aggregated_stats[0,i,:], aggregated_stats[1,i,:], np.array([fst[i]]))))
    final_stats = np.array(final_stats)
    print("final stats", final_stats.shape)
    return final_stats

def stats_per_pop(samples, benchmark=True):

    all_stats = []
    timing_data = defaultdict(list) if benchmark else None

    for i in tqdm(range(0, len(samples))):
        sample = samples[i]
        # convert -1 to 0
        sample[:, :, 0][sample[:, :, 0] == -1] = 0
        
        # Reshape sample to match expected format for stats computation
        # ss_helpers.stats_all expects shape (batch_size, num_samples, num_snps, 2)
        corrected = np.zeros((1, sample.shape[0], sample.shape[1], 2))
        corrected[0] = sample
        
        if benchmark:
            stats, timing_info = ss_helpers.stats_all(corrected, benchmark=True)
            # Store timing information
            for stat_name, timing_dict in timing_info.items():
                if isinstance(timing_dict, dict):
                    for key, value in timing_dict.items():
                        timing_data[key].append(value)
                else:
                    timing_data[stat_name].append(timing_dict)
        else:
            stats = ss_helpers.stats_all(corrected)
        
        stats_flat = list(flatten(stats))
        
        # Compute extra summary stats
        if benchmark:
            stats_extra, extra_timing = ss_extra.compute_extra_stats(sample, benchmark=True)
            # Store extra timing information
            for stat_name, timing_value in extra_timing.items():
                timing_data[stat_name].append(timing_value)
        else:
            stats_extra = ss_extra.compute_extra_stats(sample)
        
        # Concatenate all stats
        stats_all = np.concatenate([stats_flat, stats_extra])
        all_stats.append(stats_all)

    return all_stats, timing_data

def analyze_timing_statistics(timing_data):
    """
    Compute mean and standard deviation for each statistic's computation time.
    Pretty-prints the results.

    Parameters:
    -----------
    timing_data : dict
        Dictionary containing timing information for each statistic
    """
    
    timing_summary = []
    
    for stat_name, times in timing_data.items():
        if times:  # Only process if we have timing data
            times_array = np.array(times)
            summary = {
                'statistic': stat_name,
                'mean_time': np.mean(times_array),
                'std_time': np.std(times_array),
                'min_time': np.min(times_array),
                'max_time': np.max(times_array),
                'total_time': np.sum(times_array),
                'n_samples': len(times_array)
            }
            timing_summary.append(summary)
    
    timing_summary = pd.DataFrame(timing_summary)

    # Save timing data
    timing_file = f'figdata/timing_analysis.csv'
    timing_summary.to_csv(timing_file, index=False)

    print("\n" + "="*80)
    print("TIMING ANALYSIS SUMMARY")
    print("="*80)
    
    # Sort by mean time (descending)
    timing_summary_sorted = timing_summary.sort_values('mean_time', ascending=False)
    
    print(f"\n{'Statistic':<30} {'Mean (s)':<10} {'Std (s)':<10} {'Min (s)':<10} {'Max (s)':<10} {'Total (s)':<12}")
    print("-" * 92)
    
    for _, row in timing_summary_sorted.iterrows():
        print(f"{row['statistic']:<30} {row['mean_time']:<10.6f} {row['std_time']:<10.6f} "
              f"{row['min_time']:<10.6f} {row['max_time']:<10.6f} {row['total_time']:<12.6f}")
    
    
def save_stats_with_metadata(stats, max_samples=None):
    """
    Save computed statistics along with metadata.
    
    Parameters:
    -----------
    stats : np.ndarray
        Computed summary statistics
    max_samples : int, optional
        Number of samples processed (for limiting metadata updates)
    """
    # Update the original metadata file with statistics
    output_file = metadata_file()
    #metadata = pd.read_csv(output_file)

    #if max_samples is not None:
    #    metadata = metadata.head(max_samples)
    
    print("Adding statistics to metadata file...")
    assert stats.shape[1] == len(ss_helpers.ALL_STATS)
    # Add statistics columns to metadata
    metadata = {}
    for i, stat_name in enumerate(ss_helpers.ALL_STATS):
        metadata[stat_name] = stats[:, i]
    
    df = pd.DataFrame(metadata)
    df.to_csv(output_file, index=False)
    print(f"Updated metadata file: {output_file}")

    print(f"Saved statistics to metadata file")
    print(f"Statistics shape: {stats.shape}")

def main(max_samples=None, benchmark=True):    
    # Load the dataset
    print("Loading dataset...")
    samples, _ = load_data()
    print(f"Dataset shape: {samples.shape}")
    
    if max_samples is not None:
        print(f"Processing only {max_samples} samples")

    # Compute statistics
    if benchmark:
        stats, timing_data = compute_stats_for_dataset(samples, max_samples=max_samples, benchmark=True)
        
        analyze_timing_statistics(timing_data)
    else:
        stats = compute_stats_for_dataset(samples, max_samples=max_samples, benchmark=False)
    
    # Save results
    save_stats_with_metadata(stats, max_samples=max_samples)
    
if __name__ == "__main__":
    print("\nCompute statistics for the dataset.\n")   
    print("Usage: python compute_ss.py")
    main(benchmark=False)