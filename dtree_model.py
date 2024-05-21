"""
Train decision tree model on the summary stats of *real* data to predict the
output of the discriminator. Loss function is still binary cross-entropy, but
not predicting 0/1, but a probability.
Author: Sara Mathieson
Date: 5/21/24
"""

# TODO right now using MSE, but should change to binary cross-entropy

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys

def train_decision_tree(stats, disc_prob):
    """
    Train a decision tree model on the stats to predict the output of the
    discriminator
    """
    
    # split data into training and test
    X_train, X_test, y_train, y_test = train_test_split(stats, disc_prob, test_size=0.2)

    # train model
    model = DecisionTreeRegressor()
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

main()