"""
Utils for ROC curves, confusion matrix, etc
"""

import math
import numpy as np

def confusion_matrix(true, pred, prob_thresh, real=False):
    """
    predictions are logits!!
    if real=True, we only need one positive prediction per true selected region
        NOTE: not implemented yet!
    """

    conf_mat = np.array([[0,0], [0,0]]) # rows: true neutral/sel, cols: pred
    assert len(true) == len(pred)

    #prev_true_label = 0
    #print("true sel", sum(true), "true neutral", len(true)-sum(true))
    #input('enter')

    for i in range(len(true)):
        true_label = int(true[i])
        pred_label = 1 if sigmoid(pred[i]) >= prob_thresh else 0

        #if real and prev_true_label == 0 and true_label == 1: # real selected region
        #    start_checking = True
        #if start_checking:

        if true_label == 1: # true selected region
            print(sigmoid(pred[i]))
            input('enter')
            
        conf_mat[true_label, pred_label] += 1

    FPR = conf_mat[0,1]/(sum(conf_mat[0]))
    TPR = conf_mat[1,1]/(sum(conf_mat[1]))
    return FPR, TPR

def ROC(labels, predictions):
    # predictions are logits!!
    FPR_lst = []
    TPR_lst = []
    for prob_thresh in np.arange(0,1.1,0.1):
        x,y = confusion_matrix(labels, predictions, prob_thresh)
        FPR_lst.append(x)
        TPR_lst.append(y)
    
    return FPR_lst, TPR_lst

# logit -> prob
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# prob -> logit
def logit(p):
    return math.log(p/(1-p))