"""
Subsample simulated examples to match the SFS distribution of real examples,
and save as a new dataset.
"""

import os
import sys
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm

from dataset import load_data, load_metadata

def get_sfs_tuple(row, sfs_cols):
    # Returns SFS as a tuple for hashing/grouping
    return tuple(row[sfs_cols])

def main(pop, out_prefix="matched_sfs"):
    # Load data and metadata
    samples, labels = load_data(pop)
    metadata = load_metadata(pop)
    sfs_cols = [f"SFS_{i}" for i in [1, 2, 3, 4]]

    # Split real and simulated indices
    real_idx = metadata["label"] == 1
    sim_idx = metadata["label"] == 0

    # Count SFS occurrences in real data
    real_sfs = metadata.loc[real_idx, sfs_cols].apply(lambda row: get_sfs_tuple(row, sfs_cols), axis=1)
    real_sfs_counts = Counter(real_sfs)

    # For each SFS, find simulated examples with that SFS
    sim_sfs = metadata.loc[sim_idx, sfs_cols].apply(lambda row: get_sfs_tuple(row, sfs_cols), axis=1)
    sim_sfs_to_indices = defaultdict(list)
    for idx, sfs in zip(metadata.index[sim_idx], sim_sfs):
        sim_sfs_to_indices[sfs].append(idx)

    # Filter real SFS to only those present in simulated SFS
    matched_real_indices_by_sfs = defaultdict(list)
    for idx, sfs in zip(metadata.index[real_idx], real_sfs):
        if sfs in sim_sfs_to_indices:
            matched_real_indices_by_sfs[sfs].append(idx)
    total_real = sum(real_idx)
    filtered_real = sum(len(lst) for lst in matched_real_indices_by_sfs.values())
    if filtered_real < total_real:
        print(f"Dropping {total_real - filtered_real} real examples with no matching simulated SFS.")

    # For each SFS, ensure real and simulated counts match exactly
    final_real_indices = []
    matched_real_sfs_counts = {}
    for sfs, real_indices in matched_real_indices_by_sfs.items():
        sim_count = len(sim_sfs_to_indices[sfs])
        real_count = len(real_indices)
        if sim_count == 0:
            continue  # skip, should not happen due to previous filtering
        if real_count > sim_count:
            # Randomly drop real examples to match sim_count
            chosen_real = np.random.choice(real_indices, sim_count, replace=False)
            final_real_indices.extend(chosen_real)
            matched_real_sfs_counts[sfs] = sim_count
            # print(f"Dropping {real_count - sim_count} real examples for SFS {sfs} to match simulated count.")
        else:
            final_real_indices.extend(real_indices)
            matched_real_sfs_counts[sfs] = real_count

    # Subsample simulated examples to match real SFS counts (now guaranteed <= available)
    selected_sim_indices = []
    print("Subsampling...")
    for sfs, count in matched_real_sfs_counts.items():
        candidates = sim_sfs_to_indices.get(sfs, [])
        if len(candidates) >= count:
            chosen = np.random.choice(candidates, count, replace=False)
        else:
            # Should not happen now
            chosen = candidates
        selected_sim_indices.extend(chosen)

    # Combine real and selected simulated indices
    selected_real_indices = list(final_real_indices)
    selected_indices = selected_real_indices + selected_sim_indices

    # Sort indices to preserve order
    selected_indices = sorted(selected_indices)

    print(np.array(selected_indices).shape)

    # Subset samples, labels, metadata
    new_samples = samples[selected_indices]
    new_labels = labels[selected_indices]
    new_metadata = metadata.loc[selected_indices].reset_index(drop=True)

    # Save new dataset
    out_dir = f"dataset-{pop}-{out_prefix}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), new_samples)
    np.save(os.path.join(out_dir, "y.npy"), new_labels)
    new_metadata.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)
    print(f"Saved SFS-matched dataset to {out_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ab_sfs.py <population>")
        sys.exit(1)
    pop = sys.argv[1]
    main(pop)
