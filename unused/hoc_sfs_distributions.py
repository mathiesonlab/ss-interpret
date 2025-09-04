"""
Plots the distributions of SFS values (SFS_1 - SFS_9)
for real and simulated data for each population.
"""
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import sys

sys.path.insert(0, ".")
from pg_gan.ss_helpers import ALL_STATS
from utils import get_stats

def plot_one_dist(ax: Axes, real_data, simulated_data, label):

    ax.set_xlabel(label)
    ax.set_ylabel("Density")

    def p(data, color, label):
        sns.histplot(data, 
                     color=color,
                     stat='density',
                     discrete=True,
                    #  kde=True,
                     ax=ax,
                     label=label)

    p(real_data, 'blue', 'Real Data')
    p(simulated_data, 'red', 'Simulated Data')

    ax.legend()

def main():
    # Load your real and simulated data here
    populations = ["CEU", "YRI", "CHB", "CEU-matched_sfs"]

    for pop in populations:

        fig, axs = plt.subplots(3, 1, figsize=(8, 16))
        stats, real_mask = get_stats(pop, False)
        real_data = stats[real_mask]
        simulated_data = stats[~real_mask]

        for i, stat in enumerate(["SFS_1", "SFS_2", "SFS_3"]):
            sfs = ALL_STATS.index(stat)
            plot_one_dist(axs[i], real_data[:, sfs], simulated_data[:, sfs], stat)

        plt.savefig(f"./figs/sfs_{pop}.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()