"""
Compute the relationship between number of "turned on" filters and the final
prediction. Hypothesis: the more filters that are turned on, the more likely
to be under selection.
input files: (for example)
    hidden_pi_CEU_4_230410_230830_finetuneAug23_GBR.txt
    prob_CEU_4_230410_230830_finetuneAug23_GBR.txt
Author: Sara Mathieson
Date: 3/2/24
"""

# python imports
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import numpy as np
import operator
import seaborn as sns
#from sklearn.cluster import AgglomerativeClustering
import sys

################################################################################
# GLOBALS
################################################################################

#TICKS = [4.5, 26.5, 51.5, 59.5, 60.5]
#LABELS = ['SFS', 'inter-SNP distances', 'LD', '$\pi$', '#haps']
#MAX_COR = 0.8

#NUM_META_SNPS = 6 # after pooling we have this many "SNPs"
#ALL_STATS = False # if True, plot all stats, otherwise just per-SNP pi

#ABS = False # absolute value
#COLOR_MAP = {'YRI': 'PuOr', 'CEU': 'RdBu', 'CHB': 'PiYG', 'ESN': 'PuOr',
#    'GBR': 'RdBu', 'CHS': 'PiYG'}

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

        self.num_filters = len(line_dict.keys()) # num filters turned on
        
        self.hidden_data = {}
        for index in common_indices:
            if index not in line_dict:
                self.hidden_data[index] = [0.0 for _ in range(len(self.pi_vec))]
            else:
                self.hidden_data[index] = line_dict[index]

    def __str__(self):
        return str(self.pi_vec) + "\n" + str(self.hidden_data)
        

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
    return title, train

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





################################################################################
# MAIN
################################################################################

def main():
    # input and output files
    #stats_file = sys.argv[1]
    hidden_pi_file = sys.argv[1]
    pred_file = sys.argv[2]
    output_file = sys.argv[3]
    print("hidden pi file", hidden_pi_file)
    print("pred_file", hidden_pi_file)
    print("output file", output_file)

    # for plotting
    #title, train = make_title(hidden_pi_file)
    #map = COLOR_MAP[train]

    # load hidden pi
    all_hidden, common_indices = parse_correlation_file(hidden_pi_file)
    print("num regions", len(all_hidden))
    print(common_indices)

    # load predictions (probabilities)
    pred_arr = np.loadtxt(pred_file)
    print(pred_arr.shape)
    
    # plot num filters on the x-axis and prediction on the y-axis
    num_filters = [region.num_filters for region in all_hidden]
    predictions = pred_arr[:,3]
                           
    #plt.plot(num_filters, predictions, 'o', alpha=0.25, lw=0)
    #plt.show()

    # check if high scoring regions have dip in pi
    for i in range(len(all_hidden)):
        if predictions[i] > 0.9:
            print(predictions[i], all_hidden[i].pi_vec)
            input('enter')

    



if __name__ == "__main__":
    main()
