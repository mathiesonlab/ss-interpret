"""
Utils for loading models, data, weights, predictions.
Also contains base filepaths.
"""
import os
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from pg_gan.discriminator import TwoPopModel
from pg_gan.ss_helpers import ALL_STATS

BATCH_SIZE = 1000

FILTERED = ["CEU/CEU_9_230410", "CEU/CEU_12_230410", "CEU/CEU_18_230410"] + \
           [f"YRI/YRI_{i}_230410" for i in [13, 14, 15, 17, 18]] + \
           ["CHB/CHB_8_230410", "CHB/CHB_15_230410", "CHB/CHB_18_230410"] + \
           ["CEU/disc_16"] # only one failed model

SELECTED_STATS = [stat for stat in ALL_STATS if stat not in ["tajimas_d", "ones"] and 
                  not stat.startswith("inter-SNP")]

_OUTPUT_PATH = "dataset-{pop}/computed/{model_name}_preds_lf.npz"

def apply_seed_to_path(model_path: str, seed: int) -> str:
    return model_path.replace("N", str(seed))

def get_model(model_path: str, example: np.ndarray, seed: int | None = None, 
              fc_size: int = 64,
              add_norm: bool = False):
    """
    Load the discriminator model.
    """
    model = TwoPopModel(fc_size=int(fc_size), add_norm=add_norm)
    _ = model(example, training=False)

    if seed is not None:
        # model path is a template with 'N' for seed
        model.load_weights(apply_seed_to_path(model_path, seed))
    else:
        # model_path can be a specific file
        model.load_weights(model_path)

    return model

def iterate_seeds(model_path: str, start=0, stop=20):
    # skip filtered seed
    for seed in range(start, stop):
        skip = False
        for filter in FILTERED:
            if filter in apply_seed_to_path(model_path, seed):
                print(f"Skipping seed {seed} due to filter {filter}")
                skip = True
                break
        if skip:
            continue

        yield seed


def compute_all_for_dataset(model: TwoPopModel, samples: np.ndarray, max_samples=None, benchmark=True):
    if max_samples is not None:
        samples = samples[:max_samples]

    all_preds = np.zeros((len(samples), ))  
    all_lf = np.zeros((len(samples), model.fc2.units))
    timing_data = [] if benchmark else None
    
    # batched
    for i in tqdm(range(0, len(samples), BATCH_SIZE)):
        batch_samples = samples[i:i+BATCH_SIZE]
        # set first column of second channel for every sample to be 0
        # batch_samples[:, :, 0, 1] = 0.0
        # This fix has been applied; uncomment for experiment?

        # Compute predictions
        if benchmark:
            start_time = time.time()
            pass
            
        # compute predictions and output of last hidden layer
        preds = model(batch_samples, training=False)
        preds = tf.math.sigmoid(preds).numpy()[:, 0]
        hiddens = model.last_hidden_layer(batch_samples)

        # Store timing information
        if benchmark:
            timing_data.append(time.time() - start_time)

        all_preds[i:i+BATCH_SIZE] = preds
        all_lf[i:i+BATCH_SIZE] = hiddens

    if benchmark:
        return all_preds, all_lf, timing_data
    
    return all_preds, all_lf

def preds_lf_path(pop, model_str):
    return _OUTPUT_PATH.format(pop=pop, model_name=model_str)

def save_preds_lf(pop, model_str, preds, lf):
    output_dir = os.path.dirname(preds_lf_path(pop, model_str))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(preds_lf_path(pop, model_str), preds=preds, lf=lf)


def get_model_preds_lf(pop, model_name):
    """Returns a tuple (preds, learned features)"""
    a = np.load(preds_lf_path(pop, model_name))
    preds = a['preds']
    lf = a['lf']

    return preds, lf

def get_stats(pop, filter=True):
    """
    Return a tuple (stats, real_mask, valid_mask) if filtering nan ihs, or
    a tuple (stats, real_mask) if not. 
    
    Real_mask is a boolean mask for stats where the label is "real".
    """
    from dataset import load_metadata
    stats = load_metadata(pop)
    stats = stats[["source", "label"] + SELECTED_STATS]

    if filter:
        # drop weights/stats samples where stats["ihs_maxabs"] is nan
        valid_mask = ~np.isnan(stats["ihs_maxabs"])
        real_mask = (stats['label'] == 1).values
        return stats[SELECTED_STATS].values[valid_mask], real_mask[valid_mask], valid_mask
    else:
        real_mask = (stats['label'] == 1).values
        return stats[SELECTED_STATS].values, real_mask


def colorlabels(lbls, pad=0.3):
    """Color label text of the given ticklabels
    Say, from ax.get_xticklabels(0)
    """
    for label in lbls:
        text = label.get_text()
        color = None
        if "SFS" in text:
            color = "lightblue"
        elif "LD" in text:
            color = "yellow"
        elif "pi" in text or "ones" in text or "tajimas" in text:
            color = "lightgreen"
        elif "garud" in text:
            color = "lightcoral"
        elif "ihs" in text:
            color = "magenta"
        elif "haps" in text:
            color = "lightgray"
        if color:
            label.set_bbox(dict(facecolor=color, edgecolor='none', 
                                boxstyle=f'round,pad={pad}'))
