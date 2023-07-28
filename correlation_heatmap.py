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

#TICKS = [4.5, 26.5, 51.5, 59.5, 60.5]
#LABELS = ['SFS', 'inter-SNP distances', 'LD', '$\pi$', '#haps']

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

def format_function(tick, tick_pos):
    idx = TICKS.index(tick)
    return LABELS[idx]

def make_title(hidden_file):
    # train: CEU, test: GBR, seed: 2
    filename = hidden_file.split("/")[-1].split(".")[0].split("_")
    if "drex" in hidden_file:
        train = filename[2].upper()
    else:
        train = filename[1]
    test  = filename[-1]
    seed  = ''.join(c for c in filename[2] if c.isdigit())
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
    common_indices = [int(x) for x in sorted_d[0][0]] # this is the set we will use for all regions
    
    all_regions = []
    for line_group in all_groups:
        #print(line_group)
        region = RegionData(line_group, common_indices)
        all_regions.append(region)
        #print(region)
        #input("enter to continue")
    
    print("avg non-zero", np.mean([len(x)-1 for x in all_groups]))
    return all_regions, common_indices

################################################################################
# MAIN
################################################################################

def main():
    # input and output files
    correlation_file = sys.argv[1]
    #hidden_file = sys.argv[2]
    #output_file = sys.argv[3]

    print("correlation file", correlation_file)
    all_regions, common_indices = parse_correlation_file(correlation_file)
    #sys.exit()
    #print("hidden file", hidden_file)
    #print("output file", output_file)
    #title = make_title(hidden_file)

    # colormap
    #map = get_colormap(stats_file)

    #stats = np.load(stats_file)
    #stats = np.delete(stats, 0, axis=1) # remove non-seg sites since 1-pop
    #stats = np.delete(stats, 9, axis=1) # remove first inter-SNP (all zeros)

    #hidden = np.load(hidden_file)
    #print(stats.shape, hidden.shape)
    #assert stats.shape[0] == hidden.shape[0]

    #num_stats = stats.shape[1]
    #num_hidden = hidden.shape[1]

    #all_correlations = np.zeros((num_stats, num_hidden))
    #not_nan = 0
    #max_corr = 0

    for i in range(6):
            
        vec1 = [region.pi_vec[i] for region in all_regions]

        for key in common_indices:
            vec2 = [region.hidden_data[key][i] for region in all_regions] # TODO no hardcode 10!
            
            merged = np.vstack((vec1, vec2))
            print(np.corrcoef(merged)[0,1])
            if ABS:
                corr = abs(np.corrcoef(merged)[0,1]) # doing absolute value
            else:
                corr = np.corrcoef(merged)[0,1]

        '''if not math.isnan(corr):
            all_correlations[i,j] = corr
            not_nan += 1

            if abs(corr) > 0.35:
                print("corr", corr, "stat", i, "hidden", j)
                #max_corr = corr'''
        
    sys.exit()

    print(all_correlations)
    print("frac not nan:", not_nan/(num_stats*num_hidden))

    # sort columns (hidden units) by sum of their correlations
    '''all_cor_sums = corr_sum(all_correlations)
    order = np.argsort(all_cor_sums)[::-1]'''

    # sort using clustering instead
    clustering = AgglomerativeClustering().fit(np.transpose(all_correlations))
    order = order_from_children(-1, clustering.children_) # last pair 2 clusters
    all_correlations_sorted = all_correlations[:, order]

    # plot heatmap
    if ABS:
        sns.heatmap(all_correlations_sorted, vmin=0, vmax=0.5, cmap="Blues")
    else:
        ax = sns.heatmap(all_correlations_sorted, vmin=-0.5, vmax=0.5, cmap=map)

    # tick locations
    ax.yaxis.set_minor_locator(ticker.FixedLocator(TICKS))
    ax.yaxis.set_major_locator(ticker.FixedLocator([0,9,44,59,60,61]))

    # tick labels
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(format_function)

    # Remove the tick lines
    ax.tick_params(axis='y', which='minor', tick1On=False, tick2On=False)

    # align the minor tick label
    for label in ax.get_yticklabels(minor=True):
        label.set_verticalalignment('center')

    # rotate long names and space out last few
    tick_objs = ax.get_yticklabels(minor=True)
    tick_objs[0].set_rotation(90)
    tick_objs[1].set_rotation(90)
    #tick_objs[-2].set_verticalalignment('bottom')
    tick_objs[-1].set_verticalalignment('top')

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
