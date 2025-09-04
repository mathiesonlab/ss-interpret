"""
Ad-hoc: let's add a new summary statistic, one that might be correlated with some
of the other ones as well. Try a count.
"""

"""
Compute summary statistics for the dataset generated in dataset.py.
Uses logic from genome_stats.py to compute stats for both real and simulated data.
"""

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# our imports
from dataset import load_data, metadata_file

def compute_stats_for_dataset(samples):
    """
    Compute summary statistics for the entire dataset.
    
    Parameters:
    -----------
    samples : np.ndarray
        Array of shape (n_samples, num_snps, 2) containing the genomic data
        
    Returns:
    --------
    all_stats : np.ndarray
        Array of shape (n_samples, num_stats) containing computed statistics
    """
    all_stats = []
    
    for i in tqdm(range(0, len(samples))):
        sample = samples[i]
        
        # just count the number of 1s in the input
        ones = np.sum(sample[:, :, 0] == 1)

        all_stats.append(ones)
    
    return np.array(all_stats)

    
def save_stats_with_metadata(stats):
    """
    Save computed statistics along with metadata.
    
    Parameters:
    -----------
    stats : np.ndarray
        Computed summary statistics
    """
    # Update the original metadata file with statistics
    output_file = metadata_file(pop)
    metadata = pd.read_csv(output_file)

    print("Adding ones to metadata file...")
    metadata["ones"] = stats
    
    metadata.to_csv(output_file, index=False)
    print(f"Updated metadata file: {output_file}")

    print(f"Saved statistics to metadata file")

def main(pop: str):    
    # Load the dataset
    print("Loading dataset...")
    samples, _ = load_data(pop)
    print(f"Dataset shape: {samples.shape}")

    stats = compute_stats_for_dataset(samples)

    # Save results
    save_stats_with_metadata(stats)
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\nCompute 'ones' for the dataset.\n") 
        print("Example: python ad_count.py CHB")
        sys.exit(1)

    pop = sys.argv[1]
    main(pop)