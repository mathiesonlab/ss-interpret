# based on code by Ryan M. Cecil:
# https://github.com/ryanmcecil/popgen_ml_sweep_detection/blob/master/models/popgen_summary_statistics.py

import allel
import numpy as np
from typing import List

EXTRA_STATS = ['ihs_maxabs', 'tajima_d', 'garud_h1', 'garud_h12', 'garud_h2_h1', 'n_columns']

def compute_extra_stats(matrix):
    # convert our data into their format
    data = prep_our_data(matrix)

    # compute desired statistics
    ihs_maxabs = predict_ihs_max(data)
    tajima_d = predict_td(data)
    garud_h1 = predict_garud_h1(data)
    garud_h12 = predict_garud_h12(data)
    garud_h2_h1 = predict_garud_h2_h1(data)
    n_columns = predict_n_columns(data)

    stats = [ihs_maxabs, tajima_d, garud_h1, garud_h12, garud_h2_h1, n_columns]
    return stats

def prep_our_data(matrix):
    """Convert out data into the right format for functions below"""
    haplos = np.expand_dims(matrix[:,:,0], axis=-1)
    intersnp = matrix[:,:,1][0] # all the same
    positions = [sum(intersnp[:i]) for i in range(len(intersnp))]
    return [haplos, positions]

def predict_ihs_max(data: List[np.ndarray]) -> float:
    """Computes ihs statistic values from genetic data and then returns the maximum abs value of the statistics

    Parameters
    ----------
    data List[np.ndarray]: [genetic data, genetic positions]

    Returns
    -------
    output (float): Returns the statistic of the data

    """
    if not isinstance(data, list):
        raise Exception('The ihs test statistic has multiple inputs')
    genetic_data = data[0][:, :, 0]
    positions = data[1][:]
    haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int)
    h1 = allel.HaplotypeArray(haplos)
    ihs = allel.ihs(h1, pos=positions, include_edges=True)
    output = float(np.nanmax(np.abs(ihs)))
    return output

def predict_td(data: List[np.ndarray]) -> float:
    """Computes tajima d test statistic of genetic data

    Parameters
    ----------
    data (np.ndarray): Genetic data to compute test statistic of

    Returns
    -------
    output (float): Returns the statistic of the data

    """
    genetic_data = data[0][:, :, 0]
    haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int)
    h1 = allel.HaplotypeArray(haplos)
    ac = h1.count_alleles()
    output = allel.tajima_d(ac)
    return output

def predict_nsl(data: List[np.ndarray]) -> float:
    """Computes nsl test statistic of genetic data

    Parameters
    ----------
    data (np.ndarray): Genetic data to compute test statistic of

    Returns
    -------
    output (float): Returns the statistic of the data

    """
    genetic_data = data[0][:, :, 0]
    haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int)
    h1 = allel.HaplotypeArray(haplos)
    nsl = allel.nsl(h1)
    output = np.nanmax(np.abs(nsl))
    return output

def predict_garud_h1(data: List[np.ndarray]) -> float:
    """Computes garud's H1 test statistic

    Parameters
    ----------
    data (np.ndarray): Genetic data to compute test statistic of

    Returns
    -------
    output (float): Returns the statistic of the data

    """
    genetic_data = data[0][:, :, 0]
    haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int)
    h1 = allel.HaplotypeArray(haplos)
    h1, _, _, _ = allel.garud_h(h1)
    return h1

def predict_garud_h12(data: List[np.ndarray]) -> float:
    """Computes garud's H12 test statistic

    Parameters
    ----------
    data (np.ndarray): Genetic data to compute test statistic of

    Returns
    -------
    output (float): Returns the statistic of the data

    """
    genetic_data = data[0][:, :, 0]
    haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int)
    h1 = allel.HaplotypeArray(haplos)
    _, h12, _, _ = allel.garud_h(h1)
    return h12

def predict_garud_h123(data: List[np.ndarray]) -> float:
    """Computes garud's H123 test statistic

    Parameters
    ----------
    data (np.ndarray): Genetic data to compute test statistic of
    positions (Iterable): Iterable of positions if needed

    Returns
    -------
    output (float): Returns the statistic of the data

    """
    genetic_data = data[0][:, :, 0]
    haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int)
    h1 = allel.HaplotypeArray(haplos)
    _, _, h123, _ = allel.garud_h(h1)
    return h123

def predict_garud_h2_h1(data: List[np.ndarray]) -> float:
    """Computes garud's H2/H1 test statistic

    Parameters
    ----------
    data (np.ndarray): Genetic data to compute test statistic of
    positions (Iterable): Iterable of positions if needed

    Returns
    -------
    output (float): Returns the statistic of the data

    """
    genetic_data = data[0][:, :, 0]
    haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int)
    h1 = allel.HaplotypeArray(haplos)
    _, _, _, h2_h1 = allel.garud_h(h1)
    return -h2_h1  # Make negative to flip sides of classification threshold

def predict_n_columns(data: List[np.ndarray]) -> float:
    """Computes statistic based on number of columns in image

    Parameters
    ----------
    data (np.ndarray): Genetic data to compute test statistic of
    positions (Iterable): Iterable of positions if needed

    Returns
    -------
    output (float): Returns the statistic of the data

    """
    genetic_data = data[0][:, :, 0]
    return -genetic_data.shape[1]

'''
class StandardizedStatistic:
"""Implements sweep detection test statistics which are standardized in some form"""

def __init__(self,
                statistic: str):
    """Initializes the statistic

    Parameters
    ----------
    statistic: (str) - Name of the statistic
    """
    self.name = statistic
    self.threshold = None
    self.means = None
    self.stds = None
    if 'ihs_maxabs' in statistic:
        self._col_statistics = self._col_statistics_ihs
        self._predict = self._predict_ihs_max
    else:
        raise Exception(f'The test statistic {statistic} is unknown.')

def save(self,
            filename: str):
    """Saves the statistic with threshold, mean, and std values

    Parameters
    ----------
    filename: (str) - Filename that statistic settings will be saved to
    """
    if self.threshold is None:
        raise Exception("Statistic should be trained before it is saved")
    np.save(filename, self.threshold)
    np.save(f'{filename}_means', self.means)
    np.save(f'{filename}_stds', self.stds)

def load(self,
            filename: str):
    """Loads threshold, means, and stds from file

    Parameters
    ----------
    filename: (str) - Name of file that settings have been stored to
    """
    if self.threshold is not None:
        raise Exception("Threshold has already been trained or loaded")
    self.threshold = np.load(filename)
    self.means = np.load(filename.replace('.npy', '_means.npy'))
    self.stds = np.load(filename.replace('.npy', '_stds.npy'))

def set_means_stds(self,
                    means: np.ndarray,
                    stds: np.ndarray):
    """Sets the mean, stds settings for classification

    Parameters
    ----------
    means: (np.ndarray) - 1D array of bin means for standardization
    stds: (np.ndarray) - 1D array of bin standard deviations
    """
    self.means = means
    self.stds = stds

def fit(self,
        datagenerator: DataGenerator,
        **kwargs):
    """Trains the statistic model using the data supplied by the data generator and then saves it

    Parameters
    ----------
    datagenerator: (DataGenerator) - Data generator class supplying training data
    kwargs: Extra parameters that might be passed in due to Keras fit
    """
    print('============================================')
    print(f'Training Statistic {self.name} on data')

    # First process all neutral images and compute standardization settings based on derived allele frequency
    print('Computing Standardization')
    bins = [[] for _ in range(128)]
    num = 0
    for x, y in datagenerator.generator('train'):
        num += 1
        for i in range(x[0].shape[0]):
            if y[i] == 0:
                input_data = [item[i, ...] for item in x]
                ihs = self._col_statistics(input_data)
                counts = self.derived_allele_frequency(input_data[0])
                for j, count in enumerate(counts):
                    bins[count - 1].append(ihs[j])
        print(num)
    means = np.nan_to_num(np.asarray(
        [np.nanmean(binn) for binn in bins], dtype=float))
    stds = np.nan_to_num(np.asarray(
        [np.nanstd(binn) for binn in bins], dtype=float), nan=1.0)
    self.set_means_stds(means, stds)

    print('Finding Threshold')
    statistics = []
    labels = []
    # Now compute threshold
    for x, y in datagenerator.generator('train'):
        prediction = self.predict(x)
        if prediction.size != 0:
            statistics += list(prediction)
            labels += list(y)
        print(f'{len(statistics)} stats have been computed')

    self.threshold = compute_threshold(
        np.asarray(statistics), np.asarray(labels))

def predict(self, data: List[np.ndarray]) -> np.ndarray:
    stats = []
    for i in range(data[0].shape[0]):
        input_data = [item[i, ...] for item in data]
        statistic = self._predict(input_data)
        if statistic is not None:
            stats.append(statistic)
    return np.asarray(stats)

@staticmethod
def derived_allele_frequency(image_data: np.ndarray) -> np.ndarray:
    """Computes derived allele frequency for each column

    Parameters
    ----------
    image_data: (np.ndarray) - Genetic image data

    Returns
    -------
    np.ndarray: np array with allele frequency counts for each column

    """
    genetic_data = image_data[:, :, 0]
    return np.sum(genetic_data.astype(np.int), axis=0)

@staticmethod
def _col_statistics_ihs(data: List[np.ndarray]) -> np.ndarray:
    if not isinstance(data, list):
        raise Exception('The ihs test statistic has multiple inputs')
    genetic_data = data[0][:, :, 0]
    positions = data[1][:]
    haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int)
    h1 = allel.HaplotypeArray(haplos)
    ihs = allel.ihs(h1, pos=positions, include_edges=True)
    return ihs

def _predict_ihs_max(self, data: List[np.ndarray]) -> float:
    """Computes ihs statistic values from genetic data and then returns the maximum abs value of the statistics

    Parameters
    ----------
    data List[np.ndarray]: [genetic data, genetic positions]

    Returns
    -------
    output (float): Returns the statistic of the data
    """
    # Compute raw statistics
    ihs = self._col_statistics_ihs(data)
    # Standardize statistics based on bin
    counts = self.derived_allele_frequency(data[0])
    ihs = (ihs - self.means[counts - 1]) / self.stds[counts - 1]
    # Compute final prediction
    output = float(np.nanmax(np.abs(ihs)))
    if isnan(output):
        return None
    else:
        return output
'''