"""
Computes the values of hidden units of the discriminator for regions of real
data along the genome.
Authors: Sara Mathieson
Date: 7/26/23
"""

# python imports
import math
import numpy as np
import os
import scipy.special
import sys
import tensorflow as tf

# our imports
import discriminator
import global_vars
import real_data_random

################################################################################
# GLOBALS
################################################################################

NUM_SNPS = global_vars.NUM_SNPS

################################################################################
# HELPERS
################################################################################

def get_iterator(input_file, bed_file):
    iterator = real_data_random.RealDataRandomIterator(input_file, 
        global_vars.DEFAULT_SEED, bed_file)
    return iterator

# simoid
def get_prob(x):
    return 1 / (1 + math.exp(-x))

def get_pop(h5_filename):
    return h5_filename.split("/")[-1].split(".")[0]

# write a function that takes a vector and returns the indicies of non-zero values
def get_nonzero_indices(after_perm):
    num_nonz = 0
    nonz_return = None
    for row in after_perm:
        nonzero_indices = np.nonzero(row)
        num_nonz = max(num_nonz, len(nonzero_indices[0]))
        if len(nonzero_indices[0]) == num_nonz:
            nonz_return = nonzero_indices
    return nonz_return

def compute_pi(hap_matrix):
    """hap_matrix should be 2D, (num_haps, num_snps)"""
    print(hap_matrix.shape)
    num_haps = hap_matrix.shape[0]
    num_snps = hap_matrix.shape[1]
    
    # compute pi for each SNP
    num_ones = np.sum(hap_matrix, axis=0)
    num_zeros = num_haps - num_ones
    #per_snp_pi = [num_ones[i]*num_zeros[i] / scipy.special.comb(num_haps,num_ones[i]) for i in range(num_snps)]
    
    per_snp_pi = [num_ones[i]*num_zeros[i] / (num_haps*(num_haps-1)/2) for i in range(num_snps)]

    # return average pi
    return np.mean(per_snp_pi)

def homozygosity(site):
    num_indvs = len(site)//2
    homoz = 0
    for i in range(0, len(site), 2):
        if site[i] == site[i+1]:
            homoz += 1
    return homoz/num_indvs

def thetapi(hap_matrix):
    """hap_matrix should be 2D, (num_haps, num_snps)"""
    print(hap_matrix.shape)
    num_haps = hap_matrix.shape[0]
    num_snps = hap_matrix.shape[1]

    num_ones = np.sum(hap_matrix, axis=0)
    num_zeros = num_haps - num_ones

    '''per_snp_pi = []
    for s in range(num_snps):
        homoz = homozygosity(hap_matrix[:,s])
        per_snp_pi.append(sum(hap_matrix[:,s])/(1-homoz))

    return np.mean(per_snp_pi)'''

    pi = 0.0
    for i in range(num_snps):
        #homozygosity = 0.0
        homozygosity = num_zeros[i] * (num_zeros[i] - 1)
        homozygosity += num_ones[i] * (num_ones[i] - 1)   
        pi += 1.0 - homozygosity / (num_haps * (num_haps - 1))
    return pi

################################################################################
# UNPACKING THE DISCRIMINATOR
################################################################################

def disc_along_genome(iterator, input_folder, output_file=None, fine_tune_disc=None):

    if fine_tune_disc is None:
        disc = tf.saved_model.load(input_folder)
    else:
        disc = fine_tune_disc

    disc_recon = discriminator.OnePopModel(iterator.num_samples,
            saved_model=disc)

    # options for discriminator
    region_len = False
    prev_chrom = None

    # setup output array
    all_regions = []

    if output_file is not None:
        out_file = open(output_file + ".txt", 'w')

    # go through entire genome
    final_end = iterator.num_snps-NUM_SNPS
    num_total = 0
    #final_end = NUM_REGIONS*NUM_SNPS # fewer for testing
    for start_idx in range(0, final_end, NUM_SNPS):
        curr_chrom = iterator.chrom_all[start_idx]
        if curr_chrom != prev_chrom:
            print("chrom", curr_chrom, "idx", start_idx)
            prev_chrom = curr_chrom

        # get the region of real data
        # neg1 is True for discriminator and False for summary stats
        region_disc = iterator.real_region(True, region_len, start_idx=start_idx)
        region_stat = iterator.real_region(False, region_len, start_idx=start_idx)

        # compute hidden layer or probability
        if region_disc is not None:
            corrected = np.zeros((1, iterator.num_samples, NUM_SNPS, 2),
                dtype=np.float32)
            corrected[0] = region_disc

            #hidden_values = disc_recon.last_hidden_layer(corrected)
            after_perm = disc_recon.after_perm(corrected).numpy()[0]
            nonz_inds = get_nonzero_indices(after_perm)[0]
            
            for index in nonz_inds:
                to_write = str(index) + ":" + ",".join([str(h) for h in after_perm[:,index]]) + "\n"
                if output_file is not None:
                    out_file.write(to_write)
                else:
                    print(to_write)

            # look at pi in blocks of 6:
            corrected[0] = region_stat
            pi_all = compute_pi(corrected[0,:,:,0])
            theta  = thetapi(corrected[0,:,:,0])
            print("pi region:", pi_all*NUM_SNPS)
            print("thetapi", theta)
            input('enter')

            # within "meta" SNPs
            pi_vector = []
            for i in range(0,NUM_SNPS,6):
                pi = compute_pi(corrected[0,:,i:i+6,0]) # don't need inter-SNP 
                pi_vector.append(pi)
            
            to_write = "pi:" + ",".join([str(p) for p in pi_vector]) + "\n"
            if output_file is not None:
                out_file.write(to_write)
            else:
                print(to_write)
            #input('enter: got through after perm')

        num_total += 1

    print("num good regions", len(all_regions), "/", num_total) #NUM_REGIONS)
    out_file.close()

################################################################################
# MAIN
################################################################################

def test_pi():
    hap_matrix = np.array([[0,1,1,0],[1,0,1,0],[0,0,1,1],[1,1,0,0],[0,1,1,0],[0,0,1,0],[0,0,1,1],[1,1,0,0]])
    print(compute_pi(hap_matrix))

if __name__ == "__main__":
    #test_pi()
    #sys.exit()

    h5_filename = sys.argv[1]   # h5 file (i.e. real genomic regions)
    bed_filename = sys.argv[2]  # accessibility mask
    input_folder = sys.argv[3]  # folder of discriminator folders
    output_folder = sys.argv[4] # folder for npy files of hidden values
    date = sys.argv[5]

    pop = get_pop(h5_filename)
    disc_folders = sorted(os.listdir(input_folder))

    # get iterator which will return real genomic regions
    iterator = get_iterator(h5_filename, bed_filename)

    # last hidden layer or prediction for all regions
    #disc_folders = ["brooks14_exp_CEU", "brooks9_exp_CEU", "goto1_exp_CEU", "hawes13_exp_CEU",
    #    "joshi12_exp_CEU", "joshi7_exp_CEU", "rao8_exp_CEU", "sammet10_exp_CEU", "brooks4_exp_CEU",
    #    "goto11_exp_CEU", "goto6_exp_CEU", "hawes5_exp_CEU", "joshi2_exp_CEU", "rao3_exp_CEU",
    #    "sammet0_exp_CEU"]
    #disc_folders = ["brooks9_exp_YRI", "hall7_exp_YRI", "sammet5_exp_YRI", "goto6_exp_YRI", "hawes8_exp_YRI"]
    #for saved_model in disc_folders:
    for i in range(20):
        #print(disc_folders)
        '''if not HIDDEN: # only do hidden for fine-tune
            saved_model = disc_folders[0][:3] + "_" + str(i) + "_" + date
            if saved_model in disc_folders: # already trained
                input_file = input_folder + saved_model
                print("input disc", input_file)
                if HIDDEN:
                    kw = "hidden_"
                else:
                    kw = "prob_"
                output_file = output_folder + kw + saved_model + "_" + pop
                print("output file", output_file)
                if not os.path.isfile(output_file + ".txt"):
                    #print("would run predictions")
                    disc_along_genome(iterator, input_file, output_file)'''

        # fine tuning
        saved_model = disc_folders[0][:3] + "_" + str(i) + "_" + date + "_finetune"
        if saved_model in disc_folders: # already trained
            input_file = input_folder + saved_model
            print("input disc", input_file)
            kw = "hidden_pi_"
            #output_file = output_folder + kw + saved_model + "_" + pop
            #print("output file", output_file)
            disc_along_genome(iterator, input_file)#, output_file)