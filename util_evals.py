"""
Utils for ROC curves, confusion matrix, etc
"""

import math
import numpy as np

# our imports
import ss_helpers

def confusion_matrix(true, pred, prob_thresh, regions=None, real=False):
    """
    predictions are logits!!
    if real=True, we only need one positive prediction per true selected region
        NOTE: not implemented yet!
    """

    conf_mat = np.array([[0,0], [0,0]]) # rows: true neutral/sel, cols: pred
    assert len(true) == len(pred)

    stats_high_pred = []

    #prev_true_label = 0
    #print("true sel", sum(true), "true neutral", len(true)-sum(true))
    #input('enter')

    for i in range(len(true)):
        true_label = int(true[i])
        pred_label = 1 if sigmoid(pred[i]) >= prob_thresh else 0

        #if real and prev_true_label == 0 and true_label == 1: # real selected region
        #    start_checking = True
        #if start_checking:

        if prob_thresh == 0.5 and pred_label == 1: # we say selection
            print(sigmoid(pred[i]))
            print("true", true_label, "pred", pred_label)

            # compute stats
            if regions is not None:
                region = np.expand_dims(regions[i],0)
                region[region == -1] = 0 # change -1 to 0
                try:
                    stats = ss_helpers.stats_all(region)
                    stats_high_pred.append(stats[0])
                except Exception as e:
                    print(region[0,:,:,0])
                    print(e)
            #input('enter')
            
        conf_mat[true_label, pred_label] += 1

    FPR = conf_mat[0,1]/(sum(conf_mat[0]))
    TPR = conf_mat[1,1]/(sum(conf_mat[1]))
    return FPR, TPR, np.array(stats_high_pred)

def ROC(labels, predictions, regions=None):
    # predictions are logits!! (optionally pass in regions)
    FPR_lst = []
    TPR_lst = []
    print("thresholds", np.arange(0,1.1,0.1))
    for prob_thresh in np.arange(0,1.1,0.1):
        x,y,stats = confusion_matrix(labels, predictions, prob_thresh, regions=regions)
        FPR_lst.append(x)
        TPR_lst.append(y)
        if len(stats) > 0:
            np.save("stats.npy", stats)
    
    print("FPR", FPR_lst)
    print("TPR", TPR_lst)
    return FPR_lst, TPR_lst

# logit -> prob
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# prob -> logit
def logit(p):
    return math.log(p/(1-p))