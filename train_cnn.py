"""
Loads a trained discriminator (trained with pg-gan) and fine-tunes it using
simulated selection data (SLiM) with label 0 for neutral and 1 for selection.
Edits for case when discriminator is non-trained.
Authors: Sara Mathieson
Date: 8/21/23
"""

# python imports
#import math
#import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import tensorflow as tf

# our imports
#import discriminator
#import genome_disc
import global_vars
from slim_iterator import SlimIterator
#import util_evals

################################################################################
# GLOBALS
################################################################################

#SEL_TYPE = "Aug23" # change for different types of selection (i.e. Aug23, Over)
MAIN_PATH = "/homes/smathieson/Documents/pg_gan_interpret/"

# globals
TRAIN_POP = sys.argv[1] # ALL for ALL_AI
SEL_TYPE = sys.argv[2] # change for different types of selection (i.e. Aug23, Over)

#TEST_H5 = "/bigdata/smathieson/1000g-share/HDF5/" + TEST_POP + ".phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5"
#BED = "/bigdata/smathieson/1000g-share/HDF5/20120824_strict_mask.bed"
#SEL_EXCEL = MAIN_PATH + "selected_regions/selected_regions.xlsx"
SLIM_DATA = "/bigdata/smathieson/pg-gan/1000g/SLiM/Aug23/" + TRAIN_POP + "_" + SEL_TYPE + "/"

NONTRAIN_PATH = MAIN_PATH + "discriminators_og/nontrain"
#TRAIN_PATH = MAIN_PATH + "discriminators_og/" + TRAIN_POP
#PRED_PREFIX = MAIN_PATH + "discriminators_og/predictions/prob_"

NEUTRAL = "neutral"
SELECTION = ["sel_01", "sel_025", "sel_05", "sel_10"]
#if SEL_TYPE == "AI":
#    SELECTION = ["selection"]
#    #SAMPLE_SIZES = [216,198,4]
NUM_BATCH = 2000

################################################################################
# HELPERS
################################################################################

def accuracy(y_true, y_pred): # y_pred is probabilities, use 0.5 threshold
    a = 0
    assert len(y_true) == len(y_pred)
    for i in range(len(y_true)):
        logit = y_pred[i].numpy()[0]
        result = 1 if logit >= 0 else 0
        if int(y_true[i][0]) == result:
            a += 1
    return a/len(y_true)

def get_mixed_batch(batch_size, neutral_iterator, sel_iterators):
    # get a batch of half neutral data and half simulated data

    # not actually "real" data
    half = batch_size//2
    neutral_regions = neutral_iterator.real_batch(batch_size=half)
    sel_regions = mixed_sel_batch(sel_iterators, half)
    regions = np.concatenate((neutral_regions, sel_regions), axis=0)
    labels = np.concatenate((np.zeros((half,1)), np.ones((half,1))))

    # shuffle
    idx = np.random.permutation(batch_size)
    regions, labels = regions[idx], labels[idx]
   
    return regions, labels

def mixed_sel_batch(sel_iterators, num_regions):
    # num_regions should be half a batch normally
    x = len(sel_iterators) # num iters
    mini = num_regions//x
    excess = num_regions - mini*x 
    sel_regions = [iter.real_batch(batch_size=mini) for iter in sel_iterators]
    excess_regions = random.choice(sel_iterators).real_batch(batch_size=excess)
    sel_regions = np.concatenate(sel_regions + [excess_regions], axis=0)

    assert len(sel_regions) == num_regions
    return sel_regions
    
def get_test_batch(neutral_iterator, sel_iterators):

    neutral_regions = neutral_iterator.test_batch()
    num_neutral = len(neutral_regions)
    sel_regions = [iter.test_batch() for iter in sel_iterators]
    num_sel = sum([len(x) for x in sel_regions])

    regions = np.concatenate([neutral_regions] + sel_regions, axis=0)
    labels = np.concatenate((np.zeros((num_neutral,1)), np.ones((num_sel,1))))

    # don't need to shuffle
    return regions, labels

################################################################################
# FINE-TUNING
################################################################################

def fine_tune(disc_filename): #, loss_filename):#, output_filename=None):
    
    # SLiM data
    neutral_iterator = SlimIterator(SLIM_DATA + NEUTRAL)
    sel_iterators = [SlimIterator(SLIM_DATA + sel) for sel in SELECTION]

    # pg-gan trained discriminator
    disc = tf.saved_model.load(disc_filename)

    # training params
    cross_entropy =tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    # fixed test data
    test_regions, test_labels = get_test_batch(neutral_iterator, sel_iterators)
    print("test shape", test_regions.shape)

    batch_lst = []
    train_loss_lst = []
    test_loss_lst = []
    train_acc_lst = []
    test_acc_lst = []

    i = 0
    #stop = False
    while i < NUM_BATCH: # and not stop:
        with tf.GradientTape() as disc_tape:
            
            # get a mini-batch and make predictions
            train_regions, train_labels = get_mixed_batch(global_vars.BATCH_SIZE, neutral_iterator, sel_iterators)
            train_predictions = disc(train_regions, training=True)

            # evaluate loss
            train_loss = cross_entropy(train_labels, train_predictions)
            if (i % 10) == 0:
                print(i, "loss", train_loss.numpy())
                test_predictions = disc(test_regions, training=False)
                test_loss = cross_entropy(test_labels, test_predictions)
                
                # loss plot
                train_acc = accuracy(train_labels, train_predictions)
                test_acc = accuracy(test_labels, test_predictions)
                print("train acc", train_acc, "test acc", test_acc)
                
                # append to lists
                batch_lst.append(i)
                train_loss_lst.append(train_loss.numpy())
                test_loss_lst.append(test_loss.numpy())
                train_acc_lst.append(train_acc)
                test_acc_lst.append(test_acc)
                #stop = acc >= 0.95 # stop when we hit this 

            # gradient descent
            gradients_of_discriminator = disc_tape.gradient(train_loss,
                disc.trainable_variables)
            optimizer.apply_gradients(zip(gradients_of_discriminator,
                disc.trainable_variables))
        
        i += 1

    # save after fine-tuning
    #iterator = genome_disc.get_iterator(TEST_H5, BED)
    #disc_finetune = discriminator.ThreePopModel(SAMPLE_SIZES[0], SAMPLE_SIZES[1],
    #    SAMPLE_SIZES[2], saved_model=disc)
    #disc_finetune(iterator.real_batch(batch_size=1), training=False) # just to set input shapes
    
    #disc.save(disc_filename + "_finetune" + SEL_TYPE)

    # plot loss
    '''plt.clf()
    plt.plot(batch_lst, train_loss_lst, 'r--', label="train loss")
    plt.plot(batch_lst, test_loss_lst, 'r', label="test loss")
    plt.plot(batch_lst, train_acc_lst, 'b--', label="train acc")
    plt.plot(batch_lst, test_acc_lst, 'b', label="test acc")
    plt.xlabel("num batches")
    plt.legend()
    plt.savefig(loss_filename)'''

    print("num_batches = ", batch_lst)
    print("train_loss = ", train_loss_lst)

################################################################################
# MAIN
################################################################################

if __name__ == "__main__":

    #disc_folders = sorted(os.listdir(NONTRAIN_PATH))

    #for disc in disc_folders:
    for seed in range(1):

        # CEU, CHB, YRI
        disc_filename = NONTRAIN_PATH + "/nontrain_" + str(seed) + "_230821"
        #loss_filename = MAIN_PATH + "discriminators_og/figures/loss/nontrain_" + str(seed) + "_230821_loss" + str(NUM_BATCH) + ".pdf"
        
        # ALL_AI
        #disc_filename = NONTRAIN_PATH + "/AI_nontrain_" + str(seed) + "_230921"
        #loss_filename = MAIN_PATH + "discriminators_og/figures/loss/AI_nontrain_" + str(seed) + "_230921_loss" + str(NUM_BATCH) + ".pdf"

        #disc = TRAIN_PATH + "/" + TRAIN_POP + "_" + str(seed) + "_230410"

        #print(disc)
        # don't finetune agin and don't redo if we already finetuned
        # TODO change below for differnet types of selection!
        #if not ("finetune" in disc) and not ((disc + "_finetune" + SEL_TYPE) in disc_folders):
        #disc_filename = DISC_PATH + "/" + disc

        print()
        print(disc_filename)
        #print(loss_filename)
        fine_tune(disc_filename)#, loss_filename)