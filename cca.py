import numpy as np
from utils import apply_seed_to_path, colorlabels, get_model_preds_lf, get_stats, iterate_seeds
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pg_gan.ss_helpers import ALL_STATS
import pandas as pd

def one_cca_plot(pop, model_paths):
    """Generate one CCA plot for a specific model and seed."""
    stats, _, valid = get_stats(pop, filter=True)
    cca_y_weights = []
    cca_y_loadings = []
    cca_x_weights = []
    cca_x_loadings = []
    for model_path in model_paths:
        _, lf = get_model_preds_lf(pop, model_path)

        # filter out nan ihs (already done for stats)
        lf = lf[valid]

        sc = StandardScaler()
        lf_scaled = sc.fit_transform(lf)
        stats_scaled = sc.fit_transform(stats)

        # Fit CCA
        cca = CCA(n_components=2)
        lf_c, stats_c = cca.fit_transform(lf_scaled, stats_scaled)
        cca_y_weights.append(cca.y_weights_)
        cca_y_loadings.append(cca.y_loadings_)
        cca_x_weights.append(cca.x_weights_)
        cca_x_loadings.append(cca.x_loadings_)

        # # plot cca.coef_ as heatmap
        # norm = mcolors.TwoSlopeNorm(vcenter=0)
        # cg = sns.clustermap(cca.coef_, cmap="RdBu_r", norm=norm,
        #                yticklabels=ALL_STATS,
        #                xticklabels=True,
        #                row_cluster=False,
        #                )
        # cg.cax.set_position([0.1, 0.05, 0.03, 0.6])
        # cg.ax_row_dendrogram.set_visible(False)
        # cg.ax_col_dendrogram.set_visible(False)

        # cg.ax_heatmap.set_ylabel("Summary statistic")
        # cg.ax_heatmap.set_xlabel("Learned Feature")
        # plt.savefig(f"figs/cca_plot_{pop}_{model_path}.pdf", bbox_inches="tight")

        # plot 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.scatter(lf_c[:1000, 0], stats_c[:1000, 0], c="blue")
        ax1.set_xlabel("Learned feature component 1")
        ax1.set_ylabel("Summary stat component 1")
        ax1.set_title(
            f"corr = {np.corrcoef(lf_c[:1000, 0], stats_c[:1000, 0])[0, 1]:.2f}"
        )
        ax2.scatter(lf_c[:1000, 1], stats_c[:1000, 1], c="orange")
        ax2.set_xlabel("Learned feature component 2")
        ax2.set_ylabel("Summary stat component 2")
        ax2.set_title(
            f"corr = {np.corrcoef(lf_c[:1000, 1], stats_c[:1000, 1])[0, 1]:.2f}"
        )
        plt.savefig(f"figs/cca_plot_{pop}_{model_path}.pdf", bbox_inches="tight")

    # also plot loadings
    fig, axs = plt.subplots(1, len(model_paths), figsize=(6, 10))
    for i, (ax, cca_loading) in enumerate(zip(axs, cca_y_loadings)):
        cbar_on = ax == axs[-1]
        yticks = ALL_STATS if ax == axs[0] else False
        # cca_loading = np.abs(cca_loading)
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
        sns.heatmap(cca_loading, cmap="RdBu_r", norm=norm, cbar=cbar_on,
                    yticklabels=yticks, xticklabels=["CCA_1", "CCA_2"], ax=ax)
        ax.set_title(f"Seed {i}")
        # label colorbar
        if cbar_on:
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label("Loading (correlation between summary stat and CCA component)", fontsize=6)

    axs[0].set_ylabel("Summary statistic")
    for label in axs[0].get_yticklabels():
        label.set_fontsize(6)
    colorlabels(axs[0].get_yticklabels(), pad=0.1)
    plt.savefig(f"figs/cca_stat_loadings_{pop}.pdf", bbox_inches="tight")

    # and lf loadings
    fig, axs = plt.subplots(1, len(model_paths), figsize=(6, 10))
    for i, (ax, cca_loading) in enumerate(zip(axs, cca_x_loadings)):
        cbar_on = ax == axs[-1]
        yticks = (ax == axs[0])
        # cca_loading = np.abs(cca_loading)
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
        sns.heatmap(cca_loading, cmap="RdBu_r", norm=norm, cbar=cbar_on,
                    yticklabels=yticks, xticklabels=["CCA_1", "CCA_2"], ax=ax)
        ax.set_title(f"Seed {i}")

    axs[0].set_ylabel("Learned feature")
    plt.savefig(f"figs/cca_lf_loadings_{pop}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    pop = "CEU"
    models = [f"disc_{i}" for i in range(5)]
    one_cca_plot(pop, models)

    # selected_stats = [r"$\pi$", "#haps", "ihs_maxabs", "garud_h1", "garud_h12", "garud_h123", "garud_h2_h1", "SFS_1", "SFS_2", "SFS_3"] + [f"LD_{i}" for i in range(1, 15)]
    # stats, _, valid = get_stats(pop, filter=True)

    # all_metrics = []  # <-- Collect all results here

    # for model in ["initialized-N", "normal-N", "random_labels-N"]:
    #     metrics = {stat: [] for stat in selected_stats}
    #     for seed in iterate_seeds(f"discs/{pop}/{model}", stop=5):
    #         model_path = apply_seed_to_path(model, seed)
    #         _, lf = get_model_preds_lf(pop, model_path)

    #         # filter out nan ihs (already done for stats)
    #         lf = lf[valid]

    #         sc = StandardScaler()
    #         lf = sc.fit_transform(lf)
    #         stats_scaled = sc.fit_transform(stats)

    #         # Fit CCA
    #         cca = CCA(n_components=2)
    #         lf_c, stats_c = cca.fit_transform(lf, stats_scaled)

    #         for stat in selected_stats:
    #             s = stats_scaled[:, ALL_STATS.index(stat)]
    #             corrs = [np.corrcoef(s, stats_c[:, k])[0, 1] for k in range(stats_c.shape[1])]
    #             metrics[stat].append(np.max(np.abs(corrs)))

    #     metrics = {stat: np.array(corrs) for stat, corrs in metrics.items()}
    #     for stat in selected_stats:
    #         mean = np.mean(metrics[stat])
    #         std = np.std(metrics[stat])
    #         print(stat, mean, std)
    #         all_metrics.append({
    #             "model": model,
    #             "stat": stat,
    #             "mean": mean,
    #             "std": std
    #         })

    # # Save all metrics to CSV
    # df = pd.DataFrame(all_metrics)
    # df.to_csv("cca_metrics.csv", index=False)

    # df = pd.read_csv("cca_metrics.csv")
    # # plot bars
    # plt.figure(figsize=(12, 6))
    # sns.barplot(data=df, x="stat", y="mean", hue="model", errorbar="sd", palette="muted")
    # plt.xticks(rotation=45)
    # plt.title("CCA Metrics")
    # plt.tight_layout()
    # plt.savefig("cca_metrics.png", dpi=300)