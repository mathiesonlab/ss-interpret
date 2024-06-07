"""
Loads a trained discriminator (trained with pg-gan) and fine-tunes it using
simulated selection data (SLiM) with label 0 for neutral and 1 for selection.
Edits for case when discriminator is non-trained.
Authors: Sara Mathieson
Date: 8/21/23
"""

# python imports
import keras
import numpy as np
import random
import sys
import tensorflow as tf

# our imports
import discriminator
import global_vars
from slim_iterator import SlimIterator

################################################################################
# GLOBALS
################################################################################

#SEL_TYPE = "Aug23" # change for different types of selection (i.e. Aug23, Over)
#MAIN_PATH = "/homes/smathieson/Documents/pg_gan_interpret/"

# globals
TRAIN_POP = sys.argv[1] # i.e. CEU, ALL for ALL_AI
SEL_TYPE = sys.argv[2]  # change for different types of selection (i.e. Aug23, Over)

SLIM_DATA = "/bigdata/smathieson/pg-gan/1000g/SLiM/Aug23/" + TRAIN_POP + "_" + SEL_TYPE + "/"

NEUTRAL = "neutral"
SELECTION = ["sel_01", "sel_025", "sel_05", "sel_10"]
BATCH_PER_EPOCH = 200 # arbitrary

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

class SlimSequence(keras.utils.Sequence):

    def __init__(self, neutral_iterator, sel_iterators, train, batch_size=None):
        self.neutral_iterator = neutral_iterator
        self.sel_iterators = sel_iterators
        self.batch_size = batch_size # not needed for test
        self.train = train # boolean

    def __len__(self):
        """ number of batches per epoch """
        if self.train:
            return BATCH_PER_EPOCH
        return 1

    def __getitem__(self, idx):
        if self.train:
            # not actually "real" data
            half = self.batch_size//2
            neutral_regions = self.neutral_iterator.real_batch(batch_size=half)
            sel_regions = self.mixed_sel_batch(half)
            regions = np.concatenate((neutral_regions, sel_regions), axis=0)
            labels = np.concatenate((np.zeros((half,1)), np.ones((half,1))))

            # shuffle
            idx = np.random.permutation(self.batch_size)
            regions, labels = regions[idx], labels[idx]

        else:
            neutral_regions = self.neutral_iterator.test_batch()
            num_neutral = len(neutral_regions)
            sel_regions = [iter.test_batch() for iter in self.sel_iterators]
            num_sel = sum([len(x) for x in sel_regions])

            regions = np.concatenate([neutral_regions] + sel_regions, axis=0)
            labels = np.concatenate((np.zeros((num_neutral,1)), np.ones((num_sel,1))))
            # don't need to shuffle
    
        return regions, labels

    def mixed_sel_batch(self, num_regions):
        # just get selected regions
        x = len(self.sel_iterators) # num iters
        mini = num_regions//x
        excess = num_regions - mini*x 
        sel_regions = [iter.real_batch(batch_size=mini) for iter in self.sel_iterators]
        excess_regions = random.choice(self.sel_iterators).real_batch(batch_size=excess)
        sel_regions = np.concatenate(sel_regions + [excess_regions], axis=0)

        assert len(sel_regions) == num_regions
        return sel_regions

################################################################################
# TRAINING
################################################################################

def train(): #, loss_filename):#, output_filename=None):
    
    # SLiM data
    neutral_iterator = SlimIterator(SLIM_DATA + NEUTRAL)
    sel_iterators = [SlimIterator(SLIM_DATA + sel) for sel in SELECTION]

    # set up CNN model
    print("num haps", neutral_iterator.num_samples)
    model = discriminator.OnePopModel(neutral_iterator.num_samples)

    # training params
    cross_entropy =tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    # train and validation data
    training_generator = SlimSequence(neutral_iterator, sel_iterators, True, batch_size=global_vars.BATCH_SIZE)
    validation_generator = SlimSequence(neutral_iterator, sel_iterators, False)

    model.compile(optimizer=optimizer, loss=cross_entropy, metrics=['accuracy'])

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator, epochs=10)

################################################################################
# MAIN
################################################################################

if __name__ == "__main__":

    train()