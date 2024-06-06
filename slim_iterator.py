"""
Iterates through a given collection of numpy arrays representing SLiM-produced
tree sequences, in a linear order.
Interfaces both the Generator class and the RealDataRandomIterator class.

The SlimIterator accepts a .txt file containing the paths to these numpy array files.
See process_slim.py for information on generating these arrays, and below for
additional usage notes.

Files MUST be named in the pattern:
*matrices*.npy, *matrices_regions*.npy, *distances*.npy, *distances_regions*.npy,
and npy files from the same set should be listed together in the file list
(A_matrices.npy, A_distances.npy, B_matrices.npy, B_distances.npy, etc.)

The "regions" files are optional and only used for summary stats.

Author: Rebecca Riley
Date: 01/05/2023
"""

# python imports
import numpy as np
import os
import sys

# our imports
import global_vars
import util

TEST_FRAC = 0.2 # fraction used for testing

class SlimIterator:

    def __init__(self, directory):
        print("directory:", directory)
        self.matrices, self.distances, \
            matrices_regions, distances_regions = [], [], [], []

        #file_names = open(file_list, 'r').readlines()
        file_names = os.listdir(directory)

        for f in file_names:
            #file_name = file_name[:-1] # get rid of \n
            file_name = directory + "/" + f

            if "regions" in file_name:
                pass
                '''if "distances" in file_name:
                    distances_regions.append(np.load(file_name))
                elif "matrices" in file_name:
                    matrices_regions.append(np.load(file_name))'''
            elif "matrices" in file_name:
                self.matrices.append(np.load(file_name))
            elif "distances" in file_name:
                self.distances.append(np.load(file_name))
            else:
                print("warning: no match for "+file_name)

        #num_options = len(self.matrices)
        #opt_range = range(num_options)

        #self.options = np.array([i for i in opt_range])
        self.num_samples = self.matrices[0].shape[2]

        

        self.matrices = np.concatenate(self.matrices, axis=0)
        self.distances = np.concatenate(self.distances, axis=0)
        #print(self.matrices.shape)
        #print(self.distances.shape)

        # set to start after test
        self.curr_idx = int(len(self.matrices)*TEST_FRAC)

    def real_region(self, neg1, region_len=False, index=None):

        #arr_idx = self.curr_arr_idx
        if index is None: # training scenario
            index = self.curr_idx
            self.increment_indices()

        if region_len:
            gt_matrix = self.matrices_regions[index]
            dist_vec = self.distances_regions[index]

            count=0
            for i in range(len(dist_vec)):
                if dist_vec[i] != 0.0:
                    count += 1

            # print(count)
            gt_matrix, dist_vec = trim_matrix(gt_matrix, dist_vec, count)
        else:
            gt_matrix = self.matrices[index]
            dist_vec = self.distances[index]

        after = util.process_gt_dist(gt_matrix, dist_vec, region_len=region_len, neg1=neg1, real=True)

        return after

    def real_batch(self, batch_size=global_vars.BATCH_SIZE, neg1=True, region_len=False):
        """Use region_len=True for fixed region length, not by SNPs"""
        #("curr idx", self.curr_idx)

        if region_len:
            regions = [] # note: atow, all regions will be same shape, but don't actually have same # of snps due to padding!
            for i in range(batch_size):
                region = self.real_region(neg1=neg1, region_len=region_len)
                regions.append(region)

        else:
            regions = np.zeros((batch_size, self.num_samples, global_vars.NUM_SNPS, 2), dtype=np.float32)

            for i in range(batch_size):
                regions[i] = self.real_region(neg1=neg1, region_len=region_len)

        return regions
    
    def test_batch(self):
        num_test = int(len(self.matrices)*TEST_FRAC)
        regions = np.zeros((num_test, self.num_samples, global_vars.NUM_SNPS, 2), dtype=np.float32)

        for i in range(num_test):
            regions[i] = self.real_region(True, index=i) # neg 1 true

        return regions

    # interface generator too ===================================================================================
    def simulate_batch(self, batch_size=global_vars.BATCH_SIZE, neg1=True, region_len=False):
        return self.real_batch(batch_size=batch_size, neg1=neg1, region_len=region_len)

    def update_params(self, new_params):
        pass

    def increment_indices(self):
        if self.curr_idx == len(self.matrices)-1: # last index in arr
            self.curr_idx = int(len(self.matrices)*TEST_FRAC) # start after test data
        else:
            self.curr_idx += 1

def trim_matrix(gt_matrix, dist_vec, goal_snps):
    excess_size = len(dist_vec)

    half_excess = excess_size//2
    half_goal = goal_snps//2
    other_half_excess = half_excess if excess_size%2==0 else half_excess+1 # even/odd
    if goal_snps % 2 == 0 or (excess_size % 2 == 1 and goal_snps % 2 == 1):
        other_half_goal = half_goal
    else:
        other_half_goal = half_goal + 1 # even/odd

    new_matrix = gt_matrix[half_excess - half_goal : other_half_excess + other_half_goal]
    new_dist = dist_vec[half_excess - half_goal : other_half_excess + other_half_goal]

    return new_matrix, new_dist

def trim_matrix2(gt_matrix, dist_vec, goal_SNPs):
    assert type(dist_vec) == type(np.array([]))

    new_matrix = np.zeros((goal_SNPs, global_vars.DEFAULT_SAMPLE_SIZE))
    new_dist = np.zeros((goal_SNPs))

    count = 0
    for i in range(len(dist_vec)):
        if dist_vec[i] != 0.0:
            new_matrix[count] = gt_matrix[i]
            new_dist[count] = dist_vec[i]
            count+=1

    return new_matrix, new_dist

if __name__ == "__main__":
    # testing
    TRAIN_POP = sys.argv[1]
    SLIM_DATA = "/bigdata/smathieson/pg-gan/1000g/SLiM/Aug23/" + TRAIN_POP + "_Aug23/n216/"
    NEUTRAL = "neutral"

    if "AI" in SLIM_DATA:
        SELECTION = ["selection"]
    else:
        SELECTION = ["sel_01", "sel_025", "sel_05", "sel_10"]

    #neutral_iterator = SlimIterator(SLIM_DATA + NEUTRAL)
    sel_iterators = [SlimIterator(SLIM_DATA + sel) for sel in SELECTION]

    #iterator = SlimIterator(PATH)
    '''for iter in neutral_iterators:
        print(type(iter.matrices), type(iter.distances))
    for iter in sel_iterators:
        print(type(iter.matrices), type(iter.distances))'''
    #print(type(iterator.matrices))

    #test_iter = sel_iterators[0]
    #neutral_regions = neutral_iterator.test_batch()
    #print("neutral regions (all)", len(neutral_iterator.matrices))
    #print("neutral regions (test)", neutral_regions.shape)

    for iter in sel_iterators:
        regions = iter.test_batch()
        print("sel regions (all)", len(iter.matrices))
        print("sel regions (test)", regions.shape)
        #regions = test_iter.real_batch(batch_size=37, neg1=True, region_len=False)
        #print("shape:", regions.shape, "index:", test_iter.curr_idx)
