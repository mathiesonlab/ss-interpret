import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib import colormaps
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from utils import SELECTED_STATS, apply_seed_to_path, colorlabels, get_model_preds_lf, get_stats, iterate_seeds, PREFIX

def histograms_separate(preds: list[np.ndarray], 
                        name="", predsnames: list[str] = []):
    reals, simus = [], []
    for pred in preds:
        reals.append(pred[mask])
        simus.append(pred[~mask])

    def plot_hist(ax, data, bins, color, label):
        sns.histplot(
            data,
            bins=bins,
            color=color,
            label=label,
            stat="density",
            # kde=True,
            alpha=0.6,
            ax=ax,
            binrange=(0, 1)
        )

    fig, axs = plt.subplots(len(reals), 1, figsize=(10, 10), sharex=True)

    for i, (preds_real, preds_simu) in enumerate(zip(reals, simus)):
        plot_hist(axs[i], preds_real, 50, colormaps["tab10"](0), "Real Data")
        plot_hist(axs[i], preds_simu, 50, colormaps["tab10"](1), "Simulated Data")
        axs[i].set_xlim(0, 1.0)
        axs[i].legend()
        axs[i].set_xlabel("Prediction Value")
        axs[i].set_ylabel("Density")
        axs[i].set_title(f"{predsnames[i]}")

    plt.tight_layout()
    plt.savefig(f"./figs/preds_dist_{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def get_correl(weights, stats, fillna=False):
    corrs = np.zeros((weights.shape[1], stats.shape[1]))
    for i in range(weights.shape[1]):
        for j in range(stats.shape[1]):
            corrs[i, j] = np.corrcoef(weights[:, i], stats[:, j])[0, 1]

    if fillna:
        corrs_masked = np.ma.masked_invalid(corrs)
        corrs = corrs_masked.filled(0)

    corrs_df = pd.DataFrame(
        corrs, columns=SELECTED_STATS, index=[f"{i}" for i in range(weights.shape[1])]
    )

    if fillna:
        return corrs_df, corrs_masked.mask
    
    return corrs_df

def correlationclustermap(corrs_df, mask, cluster_cols=True): 
    """
    Returns a tuple (clustermap, heatmap, colorbar)
    """
    cmap = sns.color_palette("RdBu_r", as_cmap=True)
    norm = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    cmap.set_bad(color='gray')
    cg = sns.clustermap(
        corrs_df.T,
        cmap=cmap,
        norm=norm,
        mask=mask.T,
        yticklabels=True,
        xticklabels=True,
        row_cluster=False,
        col_cluster=cluster_cols,
    )

    cg.cax.set_position([0.1, 0.05, 0.03, 0.6])
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)

    cg.ax_heatmap.set_ylabel("Summary statistic")
    cg.ax_heatmap.set_xlabel("Learned Feature")

    return cg

def correlationheatmap(clustermap: sns.matrix.ClusterGrid, ax, yticks_on=False, cbar_on=False, cbar_ax=None):
    """
    Plots just the heatmap from a seaborn clustermap into the provided axis.
    Optionally plots the colorbar. 
    """
    # Extract the heatmap from the clustermap
    data = clustermap.data2d
    mask = clustermap.mask

    cmap = sns.color_palette("RdBu_r", as_cmap=True)
    norm = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    cmap.set_bad(color='gray')
    sns.heatmap(data, ax=ax,
                mask=mask,
                cbar=cbar_on,
                cbar_ax=cbar_ax,
                yticklabels=yticks_on,
                xticklabels=False,
                cmap=cmap,
                norm=norm)
    
    # ytick labels smaller
    for label in ax.get_yticklabels():
        label.set_fontsize(6)
    colorlabels(ax.get_yticklabels(), pad=0)

def plot_stacked_correlation(model_name, realgen=False):
    """
    Plots correlations for a given model. Can separate real/generated
    correlations if realgen=True.
    """
    _, weights = get_model_preds_lf(model_name)
    stats  = get_stats()

    # do we keep interSNP?
    # idxs = [.index(stat) for stat in SELECTED_STATS if stat.startswith("inter-SNP")]
    # stats = np.delete(stats, idxs, axis=1)

    # combined clustermap
    w = weights
    corrs, mask = get_correl(w, stats, fillna=True)
    cg_combined = correlationclustermap(corrs, mask, cluster_cols=True)
    col_orders = cg_combined.dendrogram_col.reordered_ind
    colorlabels(cg_combined.ax_heatmap.get_yticklabels())
    plt.savefig(PREFIX + f"figs/correl_{model_name}_combined.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    cgs = [cg_combined]
    if realgen:
        # Real and generated plots
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        for generated in [False, True]:
            
            if generated:
                w = weights[valid_mask][~real_mask]
                s = stats[~real_mask]
            else:
                w = weights[valid_mask][real_mask]
                s = stats[real_mask]
            corrs, mask = get_correl(w, s, fillna=True)

            # Reorder columns manually using stored order
            corrs_reordered = corrs.iloc[col_orders, :]
            mask_reordered = mask[col_orders, :]
            cgs.append(correlationclustermap(corrs_reordered, mask_reordered, cluster_cols=False))

        # plot all
        # fig, axs = plt.subplots(3, 1, figsize=(12, 12))
        # for i, cg in enumerate(cgs):
        #     correlationheatmap(cg, axs[i], yticks_on=True, cbar_on=True)
        #     axs[i].

        # plt.savefig(f"./figs/correl_{model_name}.pdf", dpi=300, bbox_inches="tight")

    return cgs


def plot_randomize_labels_experiment(seed: int, realgen=False):
    preds, _ = get_model_preds_lf(f"disc_{seed}")
    # preds2, _ = get_model_preds_lf(pop, f"random-labels_{seed}")
    preds3, _ = get_model_preds_lf(f"random-weights_{seed}")
    stats = get_stats()
    #preds = preds[valid_mask]
    # preds2 = preds2[valid_mask]
    #preds3 = preds3[valid_mask]

    # histogram
    #histograms_separate([preds, preds3],
    #                    name=f"exp_data_randomization_{seed}", 
    #                    predsnames=["Normal", "Random Weights"])

    # correlation
    cgs = plot_stacked_correlation(f"disc_{seed}", realgen)
    # cgs += plot_stacked_correlation(pop, f"random-labels_{seed}", realgen)
    cgs += plot_stacked_correlation(f"random-weights_{seed}", realgen)

    if len(cgs) == 2:
        # aka, just 1 row of 2 correlation maps
        fig, axs = plt.subplots(1, 2 + 1, figsize=(8, 4), width_ratios=[1, 1, 0.05])
        fig.subplots_adjust(wspace=0.05)
        for i, cg in enumerate(cgs):
            correlationheatmap(cg, axs[i], yticks_on=(i == 0), cbar_on=(i == 1),
                               cbar_ax=axs[-1] if i == 1 else None)
            axs[i].set_title(["Normal", "Random Weights"][i])
            # label x-axis
            axs[i].set_xlabel("Learned Features")

    else:
        # have to combine three stacked correlation plots into one big figure
        fig, axs = plt.subplots(3, 3, figsize=(16, 14))
        for i in range(3): # rows
            for j in range(3): # cols
                correlationheatmap(cgs[j * 3 + i], axs[i, j], yticks_on=(j == 0), cbar_on=(j == 2))
                if i == 0:
                    axs[i, j].set_title(["Normal", "Random Labels", "Random Weights"][j])
                if i == 2:
                    axs[i, j].set_xlabel("Learned Features")

    plt.savefig(f"./figs/correl_exp_random_{seed}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def linear_regression_analysis(pop, test_size=0.2, random_state=42):
    """
    Performs linear regression on activations/weights to predict summary statistics.
    Compares R² and RMSE across different models.
    """
    model_names = [f"{str}_{i}" for str in ["disc", "random-weights"] for i in range(5)]
    results = []
    
    stats, _, valid_mask = get_stats(pop)

    for model_name in model_names:
        print(f"Processing model: {model_name}")
        
        # Load data
        _, weights = get_model_preds_lf(pop, model_name)
        w, s = weights, stats

        # Apply valid mask to weights and stats
        w = w[valid_mask]
        print(f"Dropped {weights.shape[0] - w.shape[0]} samples due to nan ihsmaxabs")

        # For each summary statistic
        for stat_idx, stat_name in enumerate(SELECTED_STATS):
            y = s[:, stat_idx]
            
            # Skip if all values are the same or contain NaN
            if np.std(y) == 0 or np.any(np.isnan(y)):
                continue
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                w, y, test_size=test_size, random_state=random_state
            )
            
            # Fit linear regression
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = reg.predict(X_train)
            y_pred_test = reg.predict(X_test)
            
            # Calculate metrics
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            results.append({
                'model': model_name,
                'type': model_name.split('_')[0],
                'statistic': stat_name,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'rmse_train': rmse_train,
                'rmse_test': rmse_test,
                'n_samples': len(w),
                'n_features': w.shape[1]
            })
    
    results_df = pd.DataFrame(results)
    return results_df

def plot_dists_combined(pop):
    _, mask = get_stats(pop, filter=False)
    model_path = f"models/{pop}/{pop}_N_230410"
    for seed in iterate_seeds(model_path):
        model_name = apply_seed_to_path(model_path, seed)
        preds, _ = get_model_preds_lf(pop, model_name)# + "_corr")
        preds2, _ = get_model_preds_lf(pop, model_name)
        histograms_separate([preds, preds2], mask, name=str(seed), predsnames=["Corrected", "Uncorrected"])

    # combine all 20 of those stacked histograms into one big figure
    fig, axes = plt.subplots(5, 4, figsize=(20, 20))
    for i, ax in enumerate(axes.flat):
        img = mpimg.imread(f"./figs/preds_dist_{i}.png")
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Seed {i}")

    plt.tight_layout()
    plt.savefig(f"./figs/preds_dist_{pop}_combined.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_linear_regression_summary(results):
    # Select statistics to plot
    selected_stats = [ # "ones", 
        r"$\pi$", "#haps", "ihs_maxabs", "garud_h1", "garud_h12", "garud_h123", "garud_h2_h1",
        "SFS_1", "SFS_2", "SFS_3", "SFS_4"
    ] + [f"LD_{i}" for i in range(1, 15)]

    # Filter data
    plot_data = results[results['statistic'].isin(selected_stats)]

    # Set up the plot
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=plot_data,
        x="statistic",
        y="r2_test",
        hue="type",
        errorbar="sd",
        order=selected_stats,
        hue_order=["disc", "random-weights"],
        # legend="full",
        capsize=0.1,
        err_kws={'linewidth': 1.5}
    )
    # relabel legend
    # ax.legend(title="Model Type", labels=["Normal", "Random Weights", "Random Labels"])
    ax.set_ylabel("Test $R^2$")
    ax.set_xlabel("Summary statistic")
    # ax.set_title("Linear regression $R^2$")
    ax.set_xticks(np.arange(len(selected_stats)))
    ax.set_xticklabels(selected_stats, rotation=90)
    colorlabels(ax.get_xticklabels())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("./figs/linreg_summary.pdf", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # arguments like: pop <exp-random | corr | linreg> [seed | model name]
    parser = argparse.ArgumentParser()
    #parser.add_argument("pop", choices=["CEU", "CHB", "YRI"])
    parser.add_argument("experiment", choices=["exp-random", "corr", "linreg", "manual"])
    parser.add_argument("detail", nargs="?", default=None, 
                        help="Seed for exp-random or " \
                        "Model string like disc_0, normal-0, CEU_0_230410 for corr")
    args = parser.parse_args()

    if args.experiment == "exp-random":
        plot_randomize_labels_experiment(args.detail, False)
    elif args.experiment == "corr":
        plot_stacked_correlation(args.detail)
    elif args.experiment == "linreg":
        results = linear_regression_analysis(args.pop)
        results.to_csv("./figdata/linear_regression_analysis.csv", index=False)
        
        results = pd.read_csv("./figdata/linear_regression_analysis.csv")
        # results = results[results["model"] != "random_labels-0"]

        plot_linear_regression_summary(results)
    else:
        plot_dists_combined("CHB")