"""
Separate (simulated) pos selection results into "easy" (beginning of ROC curve)
and "hard", what features do they have?
Authors: Sara Mathieson
Date: 6/26/24
"""

# python imports
import matplotlib.pyplot as plt
import numpy as np
import os
#import random
import sklearn.metrics
import sys
import tensorflow as tf

# our imports
import discriminator
#import eval_selection
import global_vars
from slim_iterator import SlimIterator
import util_evals

MAIN_PATH = "/homes/smathieson/Documents/pg_gan_interpret/"

# globals
TRAIN_POP = sys.argv[1]
TEST_POP = sys.argv[2]
SEL_TYPE = sys.argv[3] # change for different types of selection (i.e. Aug23, Over, Over2, AI)

if SEL_TYPE == "Aug23":
    SEL_EXCEL = MAIN_PATH + "selected_regions/selected_regions.xlsx"
elif SEL_TYPE == "Over" or SEL_TYPE == "Over2":
    SEL_EXCEL = MAIN_PATH + "selected_regions/balancing_supp/S2_Table_GBE.xlsx" #_revised.xlsx"
elif SEL_TYPE == "AI":
    SEL_EXCEL = MAIN_PATH + "selected_regions/AI_supp/AI_regions_pgen.xlsx"
else:
    sys.exit("invalid selection type: " + SEL_TYPE)
ALL_GENES = MAIN_PATH + "selected_regions/refseq.allgene.txt"

if TRAIN_POP == "nontrain":
    SLIM_DATA = "/bigdata/smathieson/pg-gan/1000g/SLiM/Aug23/CEU_" + SEL_TYPE + "/"
else:
    SLIM_DATA = "/bigdata/smathieson/pg-gan/1000g/SLiM/Aug23/" + TRAIN_POP + "_" + SEL_TYPE + "/"

if TRAIN_POP == "YRI":
    SLIM_DATA += "n216/"
elif TRAIN_POP == "CHB":
    SLIM_DATA += "n206/"
    
DISC_PATH = MAIN_PATH + "discriminators_og/" + TRAIN_POP
#PRED_PREFIX = MAIN_PATH + "discriminators_og/predictions/prob_"

NEUTRAL = "neutral"
if SEL_TYPE == "AI":
    SELECTION = ["selection"]
else:
    SELECTION = ["sel_01", "sel_025", "sel_05", "sel_10"]

if TRAIN_POP == "nontrain":
    COLOR_REAL = "black"
else:
    COLOR_REAL = global_vars.COLOR_DICT[TRAIN_POP]
COLOR_SIM = "red"

################################################################################
# HELPERS
################################################################################
    
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

def plot_roc_curve(disc_before, disc_after, output_filename=None):
    plt.clf()
    
    '''
    # real data before and after
    real_predictions_before, region_data = eval_selection.all_predictions(pred_before) # our preds and regions
    real_predictions_after, region_data = eval_selection.all_predictions(pred_after) # our preds and regions
    assert len(real_predictions_before) == len(real_predictions_after)

    # real labels
    if SEL_TYPE == "Aug23":
        if TRAIN_POP == "nontrain":
            sel_region_dict = eval_selection.selected_regions(SEL_EXCEL, "CEU") # paper regions
        else:
            sel_region_dict = eval_selection.selected_regions(SEL_EXCEL, TRAIN_POP)
    elif SEL_TYPE == "Over" or SEL_TYPE == "Over2":
        sel_region_dict = eval_selection.balancing_regions(SEL_EXCEL, ALL_GENES) # TEST_POP
    elif SEL_TYPE == "AI":
        sel_region_dict = eval_selection.ai_regions(SEL_EXCEL)
    real_labels = eval_selection.get_sel_labels(region_data, sel_region_dict)
    '''
    
    # SLiM data
    neutral_iterator = SlimIterator(SLIM_DATA + NEUTRAL)
    sel_iterators = [SlimIterator(SLIM_DATA + sel) for sel in SELECTION]

    # pg-gan trained discriminator before and after fine-tuning
    seed = disc_before.split("_")[-2]
    #disc_before = tf.saved_model.load(disc_before)
    #disc_before_recon = discriminator.OnePopModel(neutral_iterator.num_samples,
    #        saved_model=disc_before)

    disc_after = tf.saved_model.load(disc_after)
    disc_after_recon = discriminator.OnePopModel(neutral_iterator.num_samples,
            saved_model=disc_after)

    # ROC before (SLiM)
    test_regions, test_labels = get_test_batch(neutral_iterator, sel_iterators)
    #print("test shape", test_regions.shape)
    #test_predictions_before = disc_before_recon(test_regions, training=False)
    #FPR_lst, TPR_lst = util_evals.ROC(test_labels, test_predictions_before)
    #auc_sim_before = round(sklearn.metrics.auc(FPR_lst, TPR_lst), 3)
    #plt.plot(FPR_lst, TPR_lst, c=COLOR_SIM, ls='--', label="SLiM, seed " + seed + ": before, AUC: " + str(auc_sim_before))
    

    # ROC after (SLiM)
    test_predictions_after = disc_after_recon(test_regions, training=False)
    FPR_lst, TPR_lst = util_evals.ROC(test_labels, test_predictions_after)
    auc_sim_after = round(sklearn.metrics.auc(FPR_lst, TPR_lst), 3)
    plt.plot(FPR_lst, TPR_lst, c=COLOR_SIM, label="SLiM, seed " + seed + ": after, AUC: " + str(auc_sim_after))

    # ROC before (real)
    '''FPR_lst, TPR_lst = util_evals.ROC(real_labels, real_predictions_before)
    auc_real_before = round(sklearn.metrics.auc(FPR_lst, TPR_lst), 3)

    # add jitter if "before" results are random
    if auc_real_before == auc_sim_before == 0.5:
        jitter = random.uniform(-0.01, 0.01)
        TPR_lst = [x+jitter for x in TPR_lst] 
    plt.plot(FPR_lst, TPR_lst, c=COLOR_REAL, ls='--', label="REAL, seed " + seed + ": before, AUC: " + str(auc_real_before))

    # ROC after (real)
    FPR_lst, TPR_lst = util_evals.ROC(real_labels, real_predictions_after)
    auc_real_after = round(sklearn.metrics.auc(FPR_lst, TPR_lst), 3)
    plt.plot(FPR_lst, TPR_lst, c=COLOR_REAL, label="REAL, seed " + seed + ": after, AUC: " + str(auc_real_after))
    '''

    # set up plot
    plt.legend()
    plt.xlim([0,1])
    plt.ylim([0,1])
    title = "ROC curve SLiM fine-tuning, train: "+TRAIN_POP+", test: "+TEST_POP
    plt.title(title)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()

    # create a ROC curve visualization
    if output_filename != None:
        plt.savefig(output_filename)
    else:
        plt.show()

################################################################################
# MAIN
################################################################################

if __name__ == "__main__":

    disc_folders = sorted(os.listdir(DISC_PATH))

    for disc in disc_folders:
        if not ("finetune" in disc) and not ("AI" in disc) and "_19_" in disc:
            if TRAIN_POP != "nontrain":
                finetune = "_230830_finetune"
            else:
                finetune = "_finetune"
            assert disc + finetune + SEL_TYPE in disc_folders # make sure we have the fine-tuning disc

            disc_before = DISC_PATH + "/" + disc
            disc_after  = DISC_PATH + "/" + disc + finetune + SEL_TYPE
            #pred_before = PRED_PREFIX + disc + "_" + TEST_POP + ".txt"
            #pred_after  = PRED_PREFIX + disc + finetune + SEL_TYPE + "_" + TEST_POP + ".txt"
            output_filename = "figs/slim"+SEL_TYPE+"_"+disc+finetune+TEST_POP+".pdf"
            
            print()
            print(disc_before)
            print(disc_after)
            #print(pred_before)
            #print(pred_after)
            print(output_filename)
            
            plot_roc_curve(disc_before, disc_after, output_filename=output_filename)
            #input('enter')