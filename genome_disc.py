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
import discriminator
import global_vars
import real_data_random

# globals
NUM_REGIONS = 1000
#SEL_TYPE = "AI" # change for different types of selection (i.e. Aug23, Over1, Over2, AI)
NUM_SNPS = global_vars.NUM_SNPS
HIDDEN = False # if True, compute last hidden layer, o.w. compute probability

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

def disc_along_genome(iterator, input_folder, output_file, fine_tune_disc=None):

    if fine_tune_disc is None:
        #disc = tf.saved_model.load(input_folder)
        #disc = tf.keras.models.load_model(input_folder)
        disc = tf.keras.layers.TFSMLayer(input_folder, call_endpoint='serving_default')
    else:
        disc = fine_tune_disc

    print("sample size", iterator.num_samples)
    disc_recon = disc #discriminator.OnePopModel(iterator.num_samples,
    #saved_model=disc)

    # options for discriminator (neg1 should be False for summary stats)
    neg1 = True
    region_len = False
    prev_chrom = None

    # setup output array
    all_regions = []
    all_logits = []

    # go through entire genome
    final_end = iterator.num_snps-NUM_SNPS
    num_total = 0
    final_end = NUM_REGIONS*NUM_SNPS # fewer for testing
    for start_idx in range(0, final_end, NUM_SNPS):
        curr_chrom = iterator.chrom_all[start_idx]
        if curr_chrom != prev_chrom:
            print("chrom", curr_chrom, "idx", start_idx)
            prev_chrom = curr_chrom

        # get the region of real data
        #print("OVERRIDING START IDX!!!!")
        #start_idx = 7651637
        #curr_chrom = iterator.chrom_all[start_idx]
        region = iterator.real_region(neg1, region_len, start_idx=start_idx)
        #print(region)

        # compute hidden layer or probability
        if region is not None:
            corrected = np.zeros((1, iterator.num_samples, NUM_SNPS, 2),
                dtype=np.float32)
            corrected[0] = region

            if HIDDEN:
                hidden_values = disc_recon.last_hidden_layer(corrected)
                all_regions.append(hidden_values.numpy()[0])

            else:
                #pred = disc(corrected, training=False).numpy()
                #print("pred", disc_recon(corrected, training=False)['output_1'].numpy())
                pred_recon = disc_recon(corrected, training=False)['output_1'].numpy()[0][0]
                #prob = get_prob(pred)
                all_logits.append(pred_recon)
                #print("logit", pred_recon)
                #input('enter')
                prob_recon = get_prob(pred_recon)

                start_base = iterator.pos_all[start_idx]
                end_idx = start_idx + global_vars.NUM_SNPS
                end_base = iterator.pos_all[end_idx]
                all_regions.append([int(curr_chrom),start_base,end_base,prob_recon])
                #print(curr_chrom,start_base,end_base,prob_recon)
                #input('enter')

        num_total += 1

    print("num good regions", len(all_regions), "/", num_total) #NUM_REGIONS)
    if HIDDEN:
        np.save(output_file + ".npy", np.array(all_regions))
    elif fine_tune_disc is None:
        f = open(output_file + ".txt", 'w')
        for row in all_regions:
            f.write("\t".join([str(x) for x in row]) + "\n")
        f.close()
    else:
        return all_logits

################################################################################
# MAIN
################################################################################

if __name__ == "__main__":

    h5_filename = sys.argv[1]   # h5 file (i.e. real genomic regions)
    bed_filename = sys.argv[2]  # accessibility mask
    input_folder = sys.argv[3]  # folder of discriminator folders
    output_folder = sys.argv[4] # folder for npy files of hidden values
    date = sys.argv[5]
    #sel_type = sys.argv[6]

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
        if not HIDDEN: # only do hidden for fine-tune
            saved_model = disc_folders[0][:3] + "_" + str(i) + "_" + date # TODO 3 or 8
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
                    disc_along_genome(iterator, input_file, output_file)

        # fine tuning
        saved_model = disc_folders[0][:3] + "_" + str(i) + "_" + date + "_finetune" # TODO 3 or 8
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
                print("would run predictions")
                #disc_along_genome(iterator, input_file, output_file)