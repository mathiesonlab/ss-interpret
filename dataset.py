"""
Utilities for loading and saving pg-gan generator and real data datasets.
"""

from __future__ import annotations # i like type annotations but py3.8 don't support :\

import io
import math
import os
import shutil
import sys

import contextlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import keras

from pg_gan.real_data_random import RealDataRandomIterator
from pg_gan.generator import Generator
from pg_gan.ss_helpers import parse_output
from pg_gan.util import parse_args, process_opts
from pg_gan.global_vars import DEFAULT_SEED, NUM_SNPS
from utils import PREFIX

OUTFILE_PATH = PREFIX + "GN-BF_gam_biallelic_2017_dadi_joint_mig_reduce_mean_filter_param3_seed1.txt"
GENOME_PATH = PREFIX + "GN-BF_gam_biallelic_2017_filter.h5"

OUTPUT_X = PREFIX + "X.npy"
OUTPUT_y = PREFIX + "y.npy"

def read_outfile(file: str) -> Generator:
    """
    Load parameters from a file, returns a generator with those params.
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} does not exist.")

    # condensed output file parsing for options. don't print
    param_values, in_file_data = parse_output(file)
    opts, param_values = parse_args(
        in_file_data=in_file_data, param_values=param_values
    )
    generator, _, _, _ = process_opts(opts, summary_stats=True)
    generator.update_params(param_values)

    print("Loaded generator:")
    print(generator.curr_params)

    return generator


def get_iterator(seed=None) -> RealDataRandomIterator:
    h5_file = GENOME_PATH
    #bed_file = BED_PATH
    s = seed if seed is not None else DEFAULT_SEED
    iterator = RealDataRandomIterator(filename=h5_file, seed=s)

    #print(f"Loaded iterator: {pop} ({s})")
    print(f"{iterator.num_snps} SNPs | {iterator.num_samples} samples")

    return iterator


def get_data(n_samples: int, seed=None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Get a dataset of real and simulated data.

    Returns a tuple of (samples, labels, sources):
        Samples is a numpy array of shape (n_samples, num_snps, 2).
        Labels is a numpy array of shape (n_samples,) with 1 for real and 0 for simulated.
        Sources is a list of strings with the source of each sample.
    """
    # iterator for real data
    iterator = get_iterator(seed=seed)

    # generator for simulated data
    outfile = OUTFILE_PATH
    # mute stdout because its a lot. i'll print params later
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        generator = read_outfile(outfile)

    # set up output data structures
    total_n = n_samples * 2
    samples = np.empty((total_n, iterator.num_samples, NUM_SNPS, 2), dtype=np.float32)
    labels = np.ones((total_n,), dtype=np.int8)

    # get n_samples from iterator
    print("Sampling from iterator")
    for i in tqdm(range(n_samples)):
        sample = iterator.real_region(neg1=True, region_len=False)
        samples[i] = sample

    # and n_samples from generator
    labels[n_samples:] = 0
    print("Sampling from generator with parameters:")
    print(generator.curr_params)

    for i in tqdm(range(n_samples)):
        sample = generator.simulate_batch(1, neg1=True, region_len=False)
        samples[n_samples + i] = sample

    assert len(samples) == len(labels)
    return samples, labels


def save_data(samples: np.ndarray, labels: np.ndarray):
    """
    Save the dataset to npz and csv files.
    """
    #d = os.path.dirname(_OUTPUT_SAMPLES.format(pop=pop))
    #if not os.path.exists(d):
    #    os.makedirs(d, exist_ok=True)

    np.save(OUTPUT_X, samples)
    np.save(OUTPUT_y, labels)

    # save sources as metadata in csv
    #metadata = pd.DataFrame({"source": sources, "label": labels})
    #metadata.to_csv(_OUTPUT_META.format(pop=pop), index=False)

    #print(f"Saved {len(samples)} samples with {samples.shape[1]} SNPs each.")
    #print(f"Metadata saved to {_OUTPUT_META.format(pop=pop)}.")


def load_data(strategy="memory", dir=None):
    """
    Load the dataset from the npz file.
    Returns a tuple of (samples, labels).
    """
    '''if dir:
        file = os.path.join(dir, f"dataset-{pop}", "samples.npz")
    else:
        file = _OUTPUT_SAMPLES.format(pop=pop)
    d = os.path.dirname(file)
    if not os.path.exists(d):
        raise FileNotFoundError(f"Dataset path {file} does not exist.")'''

    d = PREFIX

    if strategy == "memory":
        # Load samples and labels into memory
        samples = np.load(os.path.join(d, "X.npy"))
        labels = np.load(os.path.join(d, "y.npy"))
    elif strategy == "mmap":
        # copy X.npy, y.npy to /tmp (local ssd instead of NFS)
        shutil.copy(os.path.join(d, "X.npy"), "/tmp/X.npy")
        shutil.copy(os.path.join(d, "y.npy"), "/tmp/y.npy")

        samples = np.load("/tmp/X.npy", mmap_mode='r')
        labels = np.load("/tmp/y.npy", mmap_mode='r')

    print(f"Loaded {len(samples)} samples with {samples.shape[1]} SNPs each.")

    return samples, labels

def metadata_file() -> str:
    """
    Returns the path to the metadata file for the given population.
    """
    return PREFIX + "metadata.csv"

def load_metadata(pop: str) -> pd.DataFrame:
    """
    Load the metadata from the csv file.
    Returns a DataFrame with the metadata.
    """
    file = metadata_file(pop)
    if not os.path.exists(file):
        raise FileNotFoundError(f"Metadata file {file} does not exist.")

    metadata = pd.read_csv(file)
    print(f"Loaded metadata with {len(metadata)} entries.")

    return metadata


class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    From https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self, samples, labels, indices, batch_size=32, shuffle=True, seed=0, **kwargs):
        'Initialization'
        super().__init__(**kwargs)
        self.n_samples = len(indices)
        self.dim = samples.shape[1:]
        self.samples = samples
        self.labels = labels
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(self.n_samples / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        low = index * self.batch_size
        high = min(low + self.batch_size, self.n_samples)
        indices = self._indices[low:high]

        # Generate data
        X, y = self.__data_generation(self.indices[indices])

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self._indices = np.arange(self.n_samples)
        if self.shuffle:
            self.rng.shuffle(self._indices)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # can think about data augmentation here
        X = self.samples[indexes]
        y = self.labels[indexes]
        return X, y

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dataset.py n_samples") # n_samples=100000 for 100,000 each of real/sim
        sys.exit(1)

    n_samples = int(sys.argv[1])
    samples, labels = get_data(n_samples=n_samples, seed=1)
    save_data(samples, labels)
