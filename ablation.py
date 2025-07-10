# python imports
import numpy as np
import os
import sys
import tensorflow as tf
import warnings


# our imports
from pg_gan import discriminator, generator
from pg_gan import global_vars
from pg_gan import real_data_random

TRAIN_POP = "CEU"
TEST_POP = "GBR"
MODEL = "full"
SEED = 0

MODEL_PATH = f"models/{TRAIN_POP}/"
MODEL_STR = "230410.h5" if MODEL == "full" else "250626.keras"
FC_SIZE = 128 if MODEL == "full" else 64

WEIGHTS_PATH = "./results/hiddenweights/{TRAIN_POP}_{SEED}_{MODEL}_{TEST_POP}.npy"
STATS_PATH = "./data/summary_stats/stats_{TEST_POP}.npy"

def get_iterator(input_file, bed_file):
    iterator = real_data_random.RealDataRandomIterator(input_file, 
        global_vars.DEFAULT_SEED, bed_file)
    return iterator

def get_model(iterator: real_data_random.RealDataRandomIterator):
    disc = discriminator.OnePopModel(fc_size=FC_SIZE)

    # load some data to build the model
    corrected = np.zeros((1, iterator.num_samples, global_vars.NUM_SNPS, 2),
                        dtype=np.float32)
    corrected[0] = iterator.real_region(True, False)
    _ = disc(corrected, training=False)
    disc.load_weights(os.path.join(MODEL_PATH, f"{TRAIN_POP}_{SEED}_{MODEL_STR}"))

    return disc

def get_weights():
    wpath = WEIGHTS_PATH.format(
        TRAIN_POP=TRAIN_POP, SEED=SEED, MODEL=MODEL_STR.split(".")[0], TEST_POP=TEST_POP
    )
    weights = np.load(wpath)

    return weights

def get_stats():
    spath = STATS_PATH.format(TEST_POP=TEST_POP)
    stats = np.load(spath)

    return stats


def get_corrs(weights, stats, fillna=False, dropna=False):
    # warnings.filterwarnings("error", category=RuntimeWarning)

    corrs = np.zeros((weights.shape[1], stats.shape[1]))
    for i in range(weights.shape[1]):
        for j in range(stats.shape[1]):
            # try:
            corrs[i, j] = np.corrcoef(weights[:, i], stats[:, j])[0, 1]
            # except RuntimeWarning:
                # print("w:", np.mean(weights[:, i]))
                # print("s:", np.mean(stats[:, j]))

    if fillna:
        çorrs = np.ma.masked_invalid(corrs)
        # corrs = corrs_masked.filled(0)

    if dropna:
        valid_rows = ~np.all(np.isnan(corrs), axis=1)
        valid_cols = ~np.all(np.isnan(corrs), axis=0)

        corrs = corrs[valid_rows][:, valid_cols]

    return corrs


weights = get_weights()
stats = get_stats()

corrs = get_corrs(weights, stats, fillna=True)

threshold = 0.1

uncorrelated_mask = np.all(np.abs(corrs) < threshold, axis=1)

# Get the indices of such uncorrelated weights
uncorrelated_indices = np.where(uncorrelated_mask)[0]

print(f"Found {len(uncorrelated_indices)} uncorrelated weight columns (threshold={threshold}):")
print(uncorrelated_indices)

# it = get_iterator(sys.argv[1], sys.argv[2])
# model = get_model(0, it)

# print(model.layers[-1].get_weights())