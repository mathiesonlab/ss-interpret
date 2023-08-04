"""
Compute correlation between pi and after permutation-invariant function layer.
Author: Sara Mathieson
Date: 7/28/23
"""

# python imports
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import numpy as np
import operator
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import sys

################################################################################
# GLOBALS
################################################################################

TICKS = [4.5, 26.5, 51.5, 59.5, 60.5]
LABELS = ['SFS', 'inter-SNP distances', 'LD', '$\pi$', '#haps']

NUM_META_SNPS = 6 # after pooling we have this many "SNPs"

ABS = False # absolute value
COLOR_MAP = {'YRI': 'PuOr', 'CEU': 'RdBu', 'CHB': 'PiYG', 'ESN': 'PuOr',
    'GBR': 'RdBu', 'CHS': 'PiYG'}

################################################################################
# HELPERS
################################################################################

class RegionData():
    """ class to store data for a region """
    def __init__(self, lines, common_indices):
        self.pi_vec = [float(x) for x in lines[-1].split(":")[1].split(",")]

        line_dict = {}
        for line in lines[:-1]:
            line_split = line.split(":")
            i = int(line_split[0])
            line_dict[i] = [float(x) for x in line_split[1].split(",")]
        
        self.hidden_data = {}
        for index in common_indices:
            if index not in line_dict:
                self.hidden_data[index] = [0.0 for _ in range(len(self.pi_vec))]
            else:
                self.hidden_data[index] = line_dict[index]

    def __str__(self):
        #print("pi", np.average(self.pi_vec))
        #input('enter')
        return str(self.pi_vec) + "\n" + str(self.hidden_data)
        
def corr_sum(matrix):
    num_hidden = matrix.shape[1]
    return np.array([sum(matrix[:,j]) for j in range(num_hidden)])

def order_from_children(idx, child_pairs):
    """ from an array of shape (n-1, 2), determine order of columns in
    clustering.... here n=128 """

    n = len(child_pairs)+1
    left_node = child_pairs[idx][0]
    right_node = child_pairs[idx][1]

    # 4 cases (both internal nodes, both leaves, or one of each)
    if left_node >= n and right_node >= n:
        l = order_from_children(left_node-n, child_pairs)
        r = order_from_children(right_node-n, child_pairs)
        return order_from_children(left_node-n, child_pairs) + \
            order_from_children(right_node-n, child_pairs)
    elif left_node < n and right_node >= n:
        return [left_node] + order_from_children(right_node-n, child_pairs)
    elif left_node >= n and right_node < n:
        return order_from_children(left_node-n, child_pairs) + [right_node]
    else:
        return list(child_pairs[idx])

def get_colormap(stats_file):
    pop = stats_file.split("/")[-1].split("_")[1].split(".")[0]
    return COLOR_MAP[pop]

'''def xticks_format_function(tick, tick_pos):
    idx = TICKS.index(tick)
    return LABELS[idx]'''

def yticks_format_function(tick, tick_pos):
    idx = TICKS.index(tick)
    return LABELS[idx]

def make_title(hidden_file):
    # train: CEU, test: GBR, seed: 2
    filename = hidden_file.split("/")[-1].split(".")[0].split("_")
    if "drex" in hidden_file:
        train = filename[2].upper()
    else:
        train = filename[2]
    test  = filename[-1]
    seed  = ''.join(c for c in filename[3] if c.isdigit())
    title = "train: " + train + ", test: " + test + ", seed: " + seed
    return title

def parse_correlation_file(correlation_file):
    """ parse correlation file into groups of lines for each region """
    with open(correlation_file, "r") as f:
        all_data = f.readlines()

    all_groups = []
    indices_dict = defaultdict(int)
    line_group = []
    for line in all_data:
        line_group.append(line.strip())
        if line.startswith("pi"):
            # add info to relevant data structures
            all_groups.append(line_group)
            #if len(line_group) <= 3:
            indices = tuple([x.split(":")[0] for x in line_group[:-1]])
            indices_dict[indices] += 1

            # reset region
            line_group = []

    sorted_d = sorted(indices_dict.items(), key=operator.itemgetter(1),reverse=True)
    frac = sorted_d[0][1] / len(all_groups)
    common_indices = [int(x) for x in sorted_d[0][0]] # this is the set we will use for all regions
    
    all_regions = []
    for line_group in all_groups:
        region = RegionData(line_group, common_indices)
        #print(region)
        all_regions.append(region)
    
    print("avg non-zero", np.mean([len(x)-1 for x in all_groups]))
    print("common indices", common_indices, "frac", frac)
    return all_regions, common_indices

def format_yticks(ax):
    # major ticks
    ax.yaxis.set_major_locator(ticker.FixedLocator([0,9,44,59,60,61]))
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    # minor ticks
    ax.yaxis.set_minor_locator(ticker.FixedLocator(TICKS))
    ax.yaxis.set_minor_formatter(yticks_format_function)

    # Remove the tick lines
    ax.tick_params(axis='y', which='minor', tick1On=False, tick2On=False)

    # align the minor tick label
    for label in ax.get_yticklabels(minor=True):
        label.set_verticalalignment('center')

    # y-axis: rotate long stat names and space out last few
    ytick_objs = ax.get_yticklabels(minor=True)
    ytick_objs[0].set_rotation(90)
    ytick_objs[1].set_rotation(90)
    #tick_objs[-2].set_verticalalignment('bottom')
    ytick_objs[-1].set_verticalalignment('top')

def format_xticks(ax, common_indices):
    num_x = len(common_indices)*NUM_META_SNPS

    # major ticks
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(0, num_x+1, NUM_META_SNPS)))
    ax.xaxis.set_major_formatter(ticker.NullFormatter())

    # tick locations
    ax.xaxis.set_minor_locator(ticker.FixedLocator(range(NUM_META_SNPS//2, num_x, NUM_META_SNPS)))
    ax.set_xticklabels(["filter: " + str(i) for i in common_indices], minor=True)

    # Remove the tick lines
    ax.tick_params(axis='x', which='minor', tick1On=False, tick2On=False)

################################################################################
# MAIN
################################################################################

def main():
    # input and output files
    stats_file = sys.argv[1]
    hidden_pi_file = sys.argv[2]
    output_file = sys.argv[3]
    print("stats file", stats_file)
    print("hidden pi file", hidden_pi_file)
    print("output file", output_file)

    # for plotting
    title = make_title(hidden_pi_file)
    map = get_colormap(stats_file)

    # load stats
    stats = np.load(stats_file)
    stats = np.delete(stats, 0, axis=1) # remove non-seg sites since 1-pop
    stats = np.delete(stats, 9, axis=1) # remove first inter-SNP (all zeros)

    print("region 0 stat", stats[0,-2])
    print("region 1 stat", stats[1,-2])
    print("region 2 stat", stats[2,-2])
    print("region 3 stat", stats[3,-2])
    sys.exit()

    # load hidden pi
    all_hidden, common_indices = parse_correlation_file(hidden_pi_file)
    assert stats.shape[0] == len(all_hidden)
    
    # set up correlation matrix
    num_stats = stats.shape[1]
    num_hidden = NUM_META_SNPS*len(all_hidden[0].hidden_data)
    all_correlations = np.zeros((num_stats, num_hidden))
    not_nan = 0
    #max_corr = 0

    for i in range(num_stats):
        vec1 = stats[:,i]

        for j, key in enumerate(common_indices):
            for s in range(NUM_META_SNPS):
                vec2 = [region.hidden_data[key][s] for region in all_hidden]
        
                merged = np.vstack((vec1, vec2))
                #print(np.corrcoef(merged)[0,1])
                if ABS:
                    corr = abs(np.corrcoef(merged)[0,1]) # doing absolute value
                else:
                    corr = np.corrcoef(merged)[0,1]

                if not math.isnan(corr):
                    all_correlations[i,j*NUM_META_SNPS+s] = corr
                    not_nan += 1

                    if abs(corr) > 0.35:
                        print("corr", corr, "stat", i, "hidden", j)
                        #max_corr = corr

    print("frac not nan:", not_nan/(num_stats*num_hidden))

    # sort columns (hidden units) by sum of their correlations
    '''all_cor_sums = corr_sum(all_correlations)
    order = np.argsort(all_cor_sums)[::-1]'''

    # sort using clustering instead
    '''clustering = AgglomerativeClustering().fit(np.transpose(all_correlations))
    order = order_from_children(-1, clustering.children_) # last pair 2 clusters
    all_correlations_sorted = all_correlations[:, order]'''

    # plot heatmap
    if ABS:
        sns.heatmap(all_correlations, vmin=0, vmax=0.5, cmap="Blues")
    else:
        ax = sns.heatmap(all_correlations, vmin=-0.5, vmax=0.5, cmap=map)

    # plotting
    format_xticks(ax, common_indices)
    format_yticks(ax)
    plt.title(title)
    plt.tight_layout()
    #plt.show()
    plt.savefig(output_file)

def test_clustering():
    pairs = np.array([[0, 3], [1, 2], [4,5]])
    order = order_from_children(-1, pairs)
    print(order) # should be [0,3,1,2]

if __name__ == "__main__":
    #test_clustering()
    main()
