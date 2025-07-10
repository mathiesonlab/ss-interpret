import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

STATS = (
    [f"SFS_{i}" for i in range(1, 11)]
    + [f"inter-SNP_{i}" for i in range(1, 37)]
    + [f"LD_{i}" for i in range(1, 16)]
    + ["$\pi$", "#haps"]
)

EXTRA_STATS = [
    "ihs_maxabs",
    "tajimas_d",
    "garud_h1",
    "garud_h12",
    "garud_h123",
    "garud_h2_h1",
]
ALL_STATS = STATS + EXTRA_STATS

TRAIN_POP = "CEU"
TEST_POP = "GBR"
ORIG_MODEL = "230410"
REDU_MODEL = "250626"
SEEDS = list(range(0, 20))

WEIGHTS_PATH = "./results/hiddenweights/{TRAIN_POP}_{SEED}_{MODEL}_{TEST_POP}.npy"
STATS_PATH = "./data/summary_stats/stats_{TEST_POP}.npy"
PREDS_PATH = "./results/predictions/{TRAIN_POP}_{SEED}_{MODEL}_{TEST_POP}.txt"

def get_preds():
    preds_df = pd.DataFrame()

    for model, model_name in zip([ORIG_MODEL, REDU_MODEL], ["original", "reduced"]):
        # concat preds
        preds = np.array([])

        for seed in SEEDS:
            ppath = PREDS_PATH.format(
                TRAIN_POP=TRAIN_POP, SEED=seed, MODEL=model, TEST_POP=TEST_POP
            )

            preds = np.concatenate([preds, np.loadtxt(ppath)[:, -1]])

        preds_df[model_name] = preds
    
    return preds_df

def get_weights(seed, model):
    wpath = WEIGHTS_PATH.format(
        TRAIN_POP=TRAIN_POP, SEED=seed, MODEL=model, TEST_POP=TEST_POP
    )
    weights = np.load(wpath)

    return weights

def get_stats():
    spath = STATS_PATH.format(TEST_POP=TEST_POP)
    stats = np.load(spath)

    return stats

def histograms(preds_df):
    plt.figure(figsize=(8, 5))
    sns.histplot(
        preds_df["original"],
        bins=50,
        color="blue",
        label="Original",
        stat="density",
        kde=True,
        alpha=0.6,
    )
    sns.histplot(
        preds_df["reduced"],
        bins=50,
        color="orange",
        label="Reduced",
        stat="density",
        kde=True,
        alpha=0.6,
    )
    plt.legend()
    plt.xlabel("Prediction Value")
    plt.ylabel("Density")
    plt.title("Prediction distribution comparison")
    plt.savefig("./results/figs/preds_comparison.png", dpi=300, bbox_inches="tight")

def agreement(preds_df):
    pearson_corr = np.corrcoef(preds_df["original"], preds_df["reduced"])[0, 1]
    mse = mean_squared_error(preds_df["original"], preds_df["reduced"])

    print(f"Pearson correlation between original and reduced predictions: {pearson_corr:.4f}")
    print(f"Mean squared error between original and reduced predictions: {mse:.4f}")

def kde(preds_df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    # Shuffle and sample 200000 rows for KDE plot
    sampled_df = preds_df.sample(n=10000, random_state=42)
    sns.kdeplot(
        sampled_df,
        x="original",
        y="reduced",
        fill=True,
    )
    plt.xlabel("original model")
    plt.ylabel("reduced model")
    plt.title("KDE of Prediction Distributions")
    plt.savefig("./results/figs/preds_kde_comparison.png", dpi=300, bbox_inches="tight")

def get_correl(weights, stats, fillna=False):
    corrs = np.zeros((weights.shape[1], stats.shape[1]))
    for i in range(weights.shape[1]):
        for j in range(stats.shape[1]):
            corrs[i, j] = np.corrcoef(weights[:, i], stats[:, j])[0, 1]

    if fillna:
        corrs_masked = np.ma.masked_invalid(corrs)
        corrs = corrs_masked.filled(0)

    corrs_df = pd.DataFrame(
        corrs, columns=STATS, index=[f"{i}" for i in range(weights.shape[1])]
    )

    if fillna:
        return corrs_df, corrs_masked.mask
    
    return corrs_df

def correlationclustermap(corrs_df, mask): 
    cmap = sns.color_palette("RdBu", as_cmap=True)
    cmap.set_bad(color='gray')
    cg = sns.clustermap(
        corrs_df.T,
        cmap=cmap,
        center=0,
        mask=mask.T,
        cbar_kws={'label': 'Pearson Correlation'},
        yticklabels=True,
        xticklabels=True,
        row_cluster=False,
    )


    cg.cax.set_position([0.1, 0.05, 0.03, 0.6])
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)

    cg.ax_heatmap.set_ylabel("Summary statistic")
    cg.ax_heatmap.set_xlabel("Hidden activation")

    

    return cg


def correlationheatmap(corrs_df: pd.DataFrame):
    plt.figure(figsize=[12, 8])

    cg = sns.heatmap(
        corrs_df.T,
        cmap="RdBu",
        center=0
    )
    
    plt.savefig("./results/figs/correl_heatmap.png", dpi=300, bbox_inches="tight")


def mean_activation(weights):
    zero_mask = np.all(weights == 0, axis=0)
    zero_indices = np.where(zero_mask)[0]
    print(f"Indices of hidden units with zero activation across all samples: {zero_indices}")


if __name__ == "__main__":
    # preds = get_preds()
    # histograms(preds)
    # agreement(preds)
    # kde(preds)
    # mean_activation(get_weights(19, ORIG_MODEL))

    stats = get_stats()

    for model, name in zip([ORIG_MODEL, REDU_MODEL], ["original", "reduced"]):

        for seed in range(4):
            weights = get_weights(seed, model)
            corrs, mask = get_correl(weights, stats, fillna=True)
            correlationclustermap(corrs, mask)
            plt.savefig(f"./results/figs/correl_cluster_{name}_{seed}.png", dpi=300, bbox_inches="tight")


        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        for i, ax in enumerate(axes.flat):
            img = mpimg.imread(f"./results/figs/correl_cluster_{name}_{i}.png")
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Seed {i}")

        plt.tight_layout()
        plt.savefig(f"./results/figs/correl_cluster_{name}.png", dpi=300, bbox_inches="tight")
