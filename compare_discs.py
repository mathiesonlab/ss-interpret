"""
Compare all the trained discriminators on the real/all-generators dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

#from pg_gan.discriminator import OnePopModel
from dataset import load_data, load_metadata
from utils import BATCH_SIZE, apply_seed_to_path, get_model, get_model_preds_lf, iterate_seeds, PREFIX
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

model_name = "disc_0"

def main_preds():
    #for pop in pops:
    #print(f"Computing for {pop}")
    # load data
    _, labels = load_data()
    #metadata = load_metadata()

    # compute accuracy for each disc on each generator
    #metadata["source"] = metadata["source"].apply(lambda x: str(x).rsplit("_", maxsplit=1)[0])
    accs = {}
    #for model_str, fc_size in zip(model_strs, fc_sizes):
    seed = 0
    accs[seed] = {"overall": 0}
    #model_path = os.path.basename(apply_seed_to_path(model_str, seed).split(".")[0])
    preds, _ = get_model_preds_lf(model_name)
    print("got preds!", preds.shape)
    #for generator in metadata["source"].unique():
    gen_labels = labels[metadata["source"] == generator]
    gen_preds = preds[metadata["source"] == generator] > 0.5

    # store accuracy
    acc = accuracy_score(gen_labels, gen_preds)
    accs[seed][generator.split("_")[1]] = acc

    accs[seed]["overall"] += acc * (gen_preds.shape[0] / preds.shape[0])

    # metadata
    model_prefix = model_str.split("N")[0].rsplit("/")[0].replace("-", "").replace("_", "")

    # save
    accs_df = pd.DataFrame(accs).T  # rows=seeds, cols=generators
    accs_df.to_csv(PREFIX + f"figs/discriminator_accuracy_{model_prefix}.csv")

def main_viz():
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 12))
    model_prefix = model_str.split("N")[0].rsplit("/")[0].replace("-", "").replace("_", "")
    accs_df = pd.read_csv(PREFIX + f"figs/discriminator_accuracy_{model_prefix}.csv", index_col=0)
    model_prefix = "GAN" if model_prefix == "models" else "New"

    # print overall accuracy for each seed
    # for seed, acc in accs_df["overall"].items():
    #     print(f"Seed {seed:<2}: {acc:.4f}")
    print(f"Overall, {pop}, {model_prefix}: {accs_df['overall'].mean():<.3f} ({accs_df['overall'].std():.3f})")
    accs_df = accs_df.drop(columns=["overall"])

    # plt.figure(figsize=(10, 6))
    # sns.heatmap(accs_df, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1)
    # plt.xlabel("Generator")
    # plt.ylabel("Discriminator")
    # plt.title(f"{pop}")
    # plt.savefig(f"./figs/discriminator_accuracy_heatmap_{pop}_{model_prefix}.pdf", dpi=300, bbox_inches='tight')

    sns.heatmap(accs_df, annot=True, fmt=".2f", cmap="viridis", cbar=False, vmin=0, vmax=1, ax=axs[i])
    axs[i].set_xlabel("Generator")
    axs[i].set_ylabel("Discriminator")
    axs[i].set_title(f"{model_prefix}")

    fig.suptitle(
        pop,
        y=0.93,
        x=0.1,
        fontsize=16,
        bbox=dict(facecolor='white', edgecolor='black', linewidth=1)
    )
    fig.savefig(f"./figs/discriminator_accuracy_heatmap_{pop}.pdf", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # argument like python compare_discs.py <pred | fig>
    mode = sys.argv[1]
    assert mode in ["pred", "fig"], "Use 'pred' to get accuracies or 'fig' to get figures"

    if mode == "pred":
        main_preds()
    else:
        main_viz()
