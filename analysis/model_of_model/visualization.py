import matplotlib.pyplot as plt
from sklearn import tree

def linear_reg_visual(coef_lst, output_file):
    feature_dict = {}
    for i in range(len(feature_lst)):
        feature_dict[feature_lst[i]] = coef_lst[i] 

    sort_dict = sorted(feature_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_ten_feature_weights = sort_dict[:15]
    top_features, top_coeffs = zip(*top_ten_feature_weights)

    plt.figure(figsize=(9, 8))
    plt.bar(top_features, top_coeffs)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Feature")
    plt.ylabel("Coeff value")
    plt.title("Top 15 Coeff values and features")
    plt.savefig(output_file, format = 'pdf')
    plt.clf()
    plt.close()

def dtree_depth_mse(depth_lst, mse, output_file):
    plt.plot(depth_lst, mse_train, label = 'train_mse')
    plt.plot(depth_lst, mse_test, label = 'test_mse')
    plt.xlabel('depth')
    plt.ylabel('mse')
    plt.legend()
    plt.savefig(output_file, format = 'pdf')
    plt.clf()
    plt.close()

def dtree_plotting(dtree, output_file):
    plt.figure(figsize=(12,6))
    tree.plot_tree(dtree, max_depth = 4, proportion = True, feature_names = feature_lst, fontsize=20, filled = True, impurity=False, label='none')
    plt.savefig(output_file, format = 'pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

def dtree_importance(weights, output_file):
    feature_dict = {}
    for i in range(len(feature_lst)):
        feature_dict[feature_lst[i]] = weights[i] 

    sort_dict = sorted(feature_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_ten_feature_weights = sort_dict[:15]
    top_features, top_coeffs = zip(*top_ten_feature_weights)

    plt.figure(figsize=(9, 8))
    plt.bar(top_features, top_coeffs)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Feature")
    plt.ylabel("Importance value")
    plt.title("Top 15 Importance values and features")
    plt.savefig(output_file, format = 'pdf')
    plt.clf()
    plt.close()

def RF_importance(weights, output_file):
    feature_dict = {}
    for i in range(len(feature_lst)):
        feature_dict[feature_lst[i]] = weights[i] 

    sort_dict = sorted(feature_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_ten_feature_weights = sort_dict[:15]
    top_features, top_coeffs = zip(*top_ten_feature_weights)

    plt.figure(figsize=(9, 8))
    plt.bar(top_features, top_coeffs)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Feature")
    plt.ylabel("Importance value")
    plt.title("Top 15 Importance values and features")
    plt.savefig(output_file, format = 'pdf')
    plt.clf()
    plt.close()
'''
def dtree_visual(dtree, output_file='figs/tree.pdf'):
    # Export the decision tree to a dot file
    dot_data = export_graphviz(dtree, out_file=None, filled=True)
    
    # Create a graph from dot data
    graph = graphviz.Source(dot_data, format = 'pdf')
    
    graph.render(output_file, view=False)
'''

stats_orig = ['SFS0', 'SFS1', 'SFS2', 'SFS3', 'SFS4', 'SFS5', 'SFS6', 'SFS7', 'SFS8', 'SFS9', 'inter-SNP0', 'inter-SNP1', \
            'inter-SNP2', 'inter-SNP3', 'inter-SNP4', 'inter-SNP5', 'inter-SNP6', 'inter-SNP7', 'inter-SNP8', \
            'inter-SNP9', 'inter-SNP10', 'inter-SNP11', 'inter-SNP12', 'inter-SNP13', 'inter-SNP14', 'inter-SNP15',\
            'inter-SNP16', 'inter-SNP17', 'inter-SNP18', 'inter-SNP19', 'inter-SNP20', 'inter-SNP21', 'inter-SNP22', \
            'inter-SNP23', 'inter-SNP24', 'inter-SNP25', 'inter-SNP26', 'inter-SNP27', 'inter-SNP28', 'inter-SNP29', \
            'inter-SNP30', 'inter-SNP31', 'inter-SNP32', 'inter-SNP33', 'inter-SNP34', 'inter-SNP35', 'LD1', 'LD2', 'LD3',\
             'LD4', 'LD5', 'LD6', 'LD7', 'LD8', 'LD9', 'LD10', 'LD11', 'LD12', 'LD13', 'LD14', 'LD15', '$\\pi$', '#haps']

stats_extra = ['ihs_maxabs', "tajimas_d", 'garud_h1', 'garud_h12', 'garud_h123', 'garud_h2_h1']

feature_lst = stats_orig + stats_extra
