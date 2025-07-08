"""
Computes the values of the last hidden layer (or the predictions) of the
discriminator for regions of real data along the genome.
Authors: Rebecca Riley, Sara Mathieson
Date: 12/14/22
"""

# python imports
import math
import numpy as np
import os
import sys
import tensorflow as tf

# our imports
from pg_gan import discriminator
from pg_gan import global_vars
from pg_gan import real_data_random

# globals
#NUM_REGIONS = 1000
#SEL_TYPE = "AI" # change for different types of selection (i.e. Aug23, Over1, Over2, AI)
NUM_SNPS = global_vars.NUM_SNPS
HIDDEN = True # if True, compute last hidden layer, o.w. compute probability
BATCH_SIZE = 128
FC_SIZE = 128

def get_iterator(input_file, bed_file):
    iterator = real_data_random.RealDataRandomIterator(input_file, 
        global_vars.DEFAULT_SEED, bed_file)
    return iterator

# simoid
def get_prob(x):
    return 1 / (1 + math.exp(-x))

def get_pop(h5_filename):
    return h5_filename.split("/")[-1].split(".")[0]

################################################################################
# LAST HIDDEN LAYER
################################################################################

def disc_along_genome(iterator: real_data_random.RealDataRandomIterator, 
                      input_folder, output_folder, output_file,
                      fine_tune_disc: discriminator.OnePopModel=None):

    if fine_tune_disc is None:
        # load
        # disc = discriminator.OnePopModel(fc_size=128)
        disc = discriminator.OnePopModel(fc_size=FC_SIZE)

        # load some data to build the model
        corrected = np.zeros((1, iterator.num_samples, NUM_SNPS, 2),
                            dtype=np.float32)
        corrected[0] = iterator.real_region(True, False)
        _ = disc(corrected, training=False)
        disc.load_weights(input_folder)
        print(disc.layers)

        # disc = tf.keras.models.load_model(input_folder, custom_objects={"OnePopModel": discriminator.OnePopModel, "pop": 200}) # input_folder is a file in this case
    else:
        disc = fine_tune_disc

    print("sample size", iterator.num_samples)
    disc.pop = iterator.num_samples

    # options for discriminator (neg1 should be False for summary stats)
    neg1 = True
    region_len = False
    prev_chrom = None

    # setup output array
    all_regions = []
    all_hiddens = []

    # go through entire genome
    final_end = iterator.num_snps-NUM_SNPS
    num_total = 0

    # batch it
    batch_regions = []
    batch_indices = []
    for start_idx in range(0, final_end, NUM_SNPS):
        curr_chrom = iterator.chrom_all[start_idx]
        if curr_chrom != prev_chrom:
            print("chrom", curr_chrom, "idx", start_idx)
            prev_chrom = curr_chrom

        # get the region of real data
        region = iterator.real_region(neg1, region_len, start_idx=start_idx)

        # compute hidden layer or probability
        if region is not None:
            batch_regions.append(region)
            batch_indices.append(start_idx)

        if len(batch_regions) == BATCH_SIZE or start_idx + NUM_SNPS >= final_end and \
           batch_regions:
            # process batch
            corrected = np.zeros((len(batch_regions), iterator.num_samples, NUM_SNPS, 2),
                dtype=np.float32)
            for i, region in enumerate(batch_regions):
                corrected[i] = region

            if HIDDEN:
                hidden_values = disc.last_hidden_layer(corrected)
                all_hiddens.extend(hidden_values.numpy().tolist())

            pred_recon = disc(corrected, training=False)

            prob_recon = tf.math.sigmoid(pred_recon).numpy()[:, 0]

            for idx, prob in zip(batch_indices, prob_recon):
                curr_chrom = iterator.chrom_all[idx]
                start_base = iterator.pos_all[idx]
                end_idx = idx + NUM_SNPS
                end_base = iterator.pos_all[end_idx]
                all_regions.append([int(curr_chrom), start_base, end_base, prob])
                #print(curr_chrom, start_base, end_base, prob)
            
            batch_regions = []
            batch_indices = []

        num_total += 1

    print("Regions:", len(all_regions), "/", num_total)

    if HIDDEN:
        np.save(output_folder + "hiddenweights/" + output_file, np.array(all_hiddens))

    # save probs
    with open(output_folder + "predictions/" + output_file + ".txt", 'w') as f:
        for row in all_regions:
            f.write("\t".join([str(x) for x in row]) + "\n")

################################################################################
# MAIN
################################################################################

if __name__ == "__main__":

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
    #for i in range(20): # TODO change for some YRI
    #for i in [0] + list(range(8,20)):
    for i in [0]:
        #print(disc_folders)

        saved_model = disc_folders[0][:3] + "_" + str(i) + "_" + date # + ".keras"
        if saved_model in disc_folders: # already trained
            input_file = input_folder + saved_model

            print("input disc", input_file)
            output_file = saved_model.split(".")[0] + "_" + pop
            print("output file", output_file)
            disc_along_genome(iterator, input_file, output_folder, output_file)

        # fine tuning
        saved_model = disc_folders[0][:3] + "_" + str(i) + "_" + date + "_finetune" # TODO 3 or 8
        if saved_model in disc_folders: # already trained
            input_file = input_folder + saved_model
            print("input disc", input_file)
            output_file = output_folder + saved_model + "_" + pop
            print("output file", output_file)
            if not os.path.isfile(output_file + ".txt"):
                print("would run predictions")
                #disc_along_genome(iterator, input_file, output_file)