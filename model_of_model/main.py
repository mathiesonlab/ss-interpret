# python imports
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz
from scipy.stats import randint
from pathlib import Path
import os

# our imports
import visualization as viz

PATH = "/Users/smathieson/Dropbox/ss-interpret/"
#PATH = "/homes/smathieson/Documents/pg_gan_interpret/discriminators_og/"
TRAIN = "CEU" # "YRI" #"CHB" # "CEU" or nontrain
TEST = "GBR" # "ESN" #"CHS" #"GBR"

def main():

    # samples with high pred of selection (from the CNN)
    #samples_filename = "/homes/smathieson/GIT/ss-interpret/figs/stats.npy"
    #test_samples = np.load(samples_filename)

    X = np.load(PATH + "summary_stats/stats_" + TEST + "_all.npy")  
    for i in range(20):
        print("----------------")
        print(f"SEED {i}")
        output_dir = PATH + 'model_of_model/' + TRAIN + "_" + TEST + f'/seeds{i}'
        os.makedirs(output_dir, exist_ok=True)

        num = "230410"
        if TRAIN == "nontrain":
            num = "230821"
        y = read_prob_file('prob_' + TRAIN + f'_{i}_' + num + '_' + TEST + '.txt')
        X_train, y_train, X_test, y_test = train_test_split(X, y)
        
        print("Before Fine Tune")
        #print("Linear Regression results --")
        #linear_regression(X_train, y_train,  X_test, y_test, f'{output_dir}/lr_bar_chart_before.pdf')
        #naive_bayes(X, y)
        print("Decision Tree results --")
        dtree = dtree_reg(X_train, y_train,  X_test, y_test, f'{output_dir}/tree_importance_before.pdf')
        viz.dtree_plotting(dtree, f'{output_dir}/tree_before.pdf')
        #print("Random Forest results --")
        #random_forest(X_train, y_train,  X_test, y_test, f'{output_dir}/rf_importance_before.pdf')
        
        print("After Fine Tune")
        num = "230410_230830"
        if TRAIN == "nontrain":
            num = "230821"
        y = read_prob_file('prob_' + TRAIN + f'_{i}_' + num + '_finetuneAug23_' + TEST + '.txt')
        X_train, y_train,  X_test, y_test = train_test_split(X, y)
        #print("\nLinear Regression results --")
        #linear_regression(X_train, y_train,  X_test, y_test, f'{output_dir}/lr_bar_chart_after.pdf')
        print("Decision Tree results --")
        dtree = dtree_reg(X_train, y_train,  X_test, y_test, f'{output_dir}/tree_importance_after.pdf')
        #decision_paths = dtree.decision_path(test_samples)
        #leaf_id = dtree.apply(test_samples)
        #print_path(decision_paths, leaf_id, test_samples, dtree)
        #print(decision_paths)
        #input('enter')
        viz.dtree_plotting(dtree, f'{output_dir}/tree_after.pdf')
        #print("Random Forest results --")
        #random_forest(X_train, y_train,  X_test, y_test, f'{output_dir}/rf_importance_after.pdf')

def read_prob_file (filename):
    y = []
    with open (PATH + f'predictions/{filename}', 'r') as file:
        lines = file.readlines()
    for line in lines:
        y.append(float(line.split()[-1])) # append the last entry of one line
    return y

def train_test_split(X, y):
    num_samples = len(y)
    index_lst = np.arange(num_samples)
    np.random.shuffle(index_lst)
    num_train = int(num_samples * 0.8)
    y = np.array(y)

    X_train = X[index_lst[: num_train]]
    X_test = X[index_lst[num_train: ]]
    y_train = y[index_lst[: num_train]]
    y_test = y[index_lst[num_train: ]]
    return X_train, y_train,  X_test, y_test


'''def linear_regression(X_train, y_train, X_test, y_test, output_file):
    reg = LinearRegression().fit(X_train, y_train)
    coef_lst = reg.coef_
    pred_y_train = reg.predict(X_train)
    pred_y_test = reg.predict(X_test)
    print(f"MSE for train: {mean_squared_error(y_train, pred_y_train)}")
    print(f"MSE for test: {mean_squared_error(y_test, pred_y_test)}")
    viz.linear_reg_visual(coef_lst, output_file)        
    return coef_lst'''

def dtree_reg(X_train, y_train, X_test, y_test, output_file):
    mse_train = []
    mse_test = []
    dtree_lst = []
    depth_lst = [9] #[1,3,5,7,9,11,13,15]#i + 1 for i in range(10)]
    for depth in depth_lst:
        dtree = tree.DecisionTreeRegressor(max_depth = depth).fit(X_train, y_train)
        dtree_lst.append(dtree)
        pred_y_train = dtree.predict(X_train)
        pred_y_test = dtree.predict(X_test)
        mse_train.append(mean_squared_error(y_train, pred_y_train))
        mse_test.append(mean_squared_error(y_test, pred_y_test))

        # r^2
        r2_test = r2_score(y_test, pred_y_test)
        print("r2", r2_test, "depth", depth)
        plt.clf()
        plt.scatter(y_test, pred_y_test, s=5, color="cornflowerblue")
        plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, pred_y_test, 1))(np.unique(y_test)), color="darkorange")
        plt.axis([0.47,0.95,0.47,0.95])
        plt.title(f"$r^2$: {round(r2_test,3)}, Decision tree depth: {depth}", fontsize=20)
        plt.xlabel("CNN prediction", fontsize=16)
        plt.ylabel("Decision Tree prediction", fontsize=16)
        plt.show()

    idx = 0 #np.argmin(mse_test) TODO put back, just using depth 3 for viz or 9 for final
    min_depth = depth_lst[idx]# + 1
    print(f"Min test mse: {np.min(mse_test)}; Depth: {min_depth}")
    print(f"Corresponding train mse: {mse_train[idx]}")
    weights = dtree_lst[idx].feature_importances_
    #viz.dtree_importance(weights, output_file)
    return dtree_lst[idx]

def print_path(node_indicator, leaf_id, X_test, clf):
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    sample_id = 200
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
    ]

    print("Rules used to predict sample {id}:\n".format(id=sample_id))
    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == node_id:
            continue

        # check if value of the split feature for sample 0 is below threshold
        if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print(
            "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
            "{inequality} {threshold})".format(
                node=node_id,
                sample=sample_id,
                feature=feature[node_id],
                value=X_test[sample_id, feature[node_id]],
                inequality=threshold_sign,
                threshold=threshold[node_id],
            )
        )

'''
def dtree_reg(X_train, y_train, X_test, y_test, output_file):
    #Find the optimal hyperparameters for decision tree
    dtree_reg = tree.DecisionTreeRegressor(random_state=42)

    param_dist = {
        'max_depth': randint(6, 10),  # Focusing around the optimal values found
        'min_samples_split': randint(3, 20), 
        'min_samples_leaf': randint(4, 12)
    }

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(dtree_reg, param_distributions=param_dist, 
                                    n_iter=100, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)

    # Fit the model
    random_search.fit(X_train, y_train)

    # Get the best parameters and best score
    best_params_random = random_search.best_params_
    best_score_random = random_search.best_score_

    print(f"Best Parameters (Random Search): {best_params_random}")
    print(f"Best Score (Random Search): {best_score_random}")

    # Evaluate the model on the test set
    best_model_random = random_search.best_estimator_
    y_pred_random = best_model_random.predict(X_test)
    mse_random = mean_squared_error(y_test, y_pred_random)
    print(f"Test Set MSE (Random Search): {mse_random}")
'''

'''
def dtree_path(dtree, X_train, X_test):
    X = np.concatenate((X_train, X_test), axis = 0)
    pred_y = dtree.predict(X)
    min_pred = np.min(pred_y)
    min_pred_X = X[np.argmin(pred_y)]
    max_pred_X = X[np.argmax(pred_y)]
    decision_path = dtree.apply(min_pred_X)
'''

'''def naive_bayes(X, y):
    y = np.array(y).reshape(1, -1)
    y_bin = Binarizer(threshold = 0.5).fit(y)
    y = y_bin.transform(y)[0]
    print(len(y))
    
    clf = GaussianNB().fit(X, y)
    pred_y = clf.predict_proba(X)
    print(pred_y)
    print(f"MSE: {mean_squared_error(y, pred_y[:,1])}")

def random_forest(X_train, y_train, X_test, y_test, output_file):
    classifier_rf = RandomForestRegressor(random_state=42, max_depth= 7, n_estimators=100, oob_score=True)
    classifier_rf.fit(X_train, y_train)
    pred_y_train = classifier_rf.predict(X_train)
    pred_y_test = classifier_rf.predict(X_test)
    print(f"MSE for train: {mean_squared_error(y_train, pred_y_train)}")
    print(f"MSE for test: {mean_squared_error(y_test, pred_y_test)}")
    importances = classifier_rf.feature_importances_
    viz.RF_importance(importances, output_file)

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    params = {
        'max_depth': [2,3,5,10,20],
        'min_samples_leaf': [5,10,20,50,100,200],
        'n_estimators': [10,25,30,50,100,200]
    }
    grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="neg_mean_squared_error")

    grid_search.fit(X_train, y_train)

    rf_best = grid_search.best_estimator_
    print(grid_search.best_score_)
    print(rf_best)'''


feature_lst = ['SFS0', 'SFS1', 'SFS2', 'SFS3', 'SFS4', 'SFS5', 'SFS6', 'SFS7', 'SFS8', 'SFS9', 'inter-SNP0', 'inter-SNP1', \
            'inter-SNP2', 'inter-SNP3', 'inter-SNP4', 'inter-SNP5', 'inter-SNP6', 'inter-SNP7', 'inter-SNP8', \
            'inter-SNP9', 'inter-SNP10', 'inter-SNP11', 'inter-SNP12', 'inter-SNP13', 'inter-SNP14', 'inter-SNP15',\
            'inter-SNP16', 'inter-SNP17', 'inter-SNP18', 'inter-SNP19', 'inter-SNP20', 'inter-SNP21', 'inter-SNP22', \
            'inter-SNP23', 'inter-SNP24', 'inter-SNP25', 'inter-SNP26', 'inter-SNP27', 'inter-SNP28', 'inter-SNP29', \
            'inter-SNP30', 'inter-SNP31', 'inter-SNP32', 'inter-SNP33', 'inter-SNP34', 'inter-SNP35', 'LD1', 'LD2', 'LD3',\
             'LD4', 'LD5', 'LD6', 'LD7', 'LD8', 'LD9', 'LD10', 'LD11', 'LD12', 'LD13', 'LD14', 'LD15', '$\\pi$', '#haps']

main()