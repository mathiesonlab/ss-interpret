"""
Train decision tree model on the summary stats of *real* data to predict the
output of the discriminator. Not back-prop so there is no loss function, but
using regression to predict the probability of selection directly.
Author: Sara Mathieson
Date: 5/21/24
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys

NUM_SFS = 9
NUM_INTERSNP = 35
NUM_LD = 15
SFS_NAMES = ['SFS'+str(i) for i in range(1,NUM_SFS+1)]
INTER_SNP_NAMES = ['inter-SNP'+str(i) for i in range(1,NUM_INTERSNP+1)]
LD_NAMES = ['LD'+str(i) for i in range(1,NUM_LD+1)]
SS_NAMES = SFS_NAMES + INTER_SNP_NAMES + LD_NAMES + ['$\pi$', '#haps']

def train_decision_tree(stats, disc_prob):
    """
    Train a decision tree model on the stats to predict the output of the
    discriminator
    """

    # split data into training and test
    X_train, X_test, y_train, y_test = train_test_split(stats, disc_prob, test_size=0.2)

    # train model
    model = DecisionTreeRegressor(max_depth=10)
    model.fit(X_train, y_train)

    # evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean squared error:", mse)

    return model

def main():
    # input and output files
    stats_file = sys.argv[1]
    disc_pred_file = sys.argv[2]
    print("stats file", stats_file)
    print("disc pred file", disc_pred_file)

    # for plotting
    #title, train = make_title(hidden_pi_file)
    #map = COLOR_MAP[train]

    # load stats
    stats = np.load(stats_file)
    stats = np.delete(stats, 0, axis=1) # remove non-seg sites since 1-pop
    stats = np.delete(stats, 9, axis=1) # remove first inter-SNP (all zeros)
    print(stats.shape)

    disc_prob = np.loadtxt(disc_pred_file)[:,-1]
    print(disc_prob.shape)

    assert stats.shape[0] == len(disc_prob)

    # train model
    dtree_model = train_decision_tree(stats, disc_prob)
    print(dtree_model.feature_importances_)

    tree.plot_tree(dtree_model, feature_names=SS_NAMES, filled=True)
    plt.savefig("figs/dtree_model.pdf")

main()
