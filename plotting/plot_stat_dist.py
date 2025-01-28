import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

EXTRA_STATS = ['ihs_maxabs', "Tajima's D", 'garud_h1', 'garud_h12', 'garud_h123', 'garud_h2_h1']
MAX_VALUE = 1000

# Load the .npy file
stats = np.load(sys.argv[1])
preds = np.loadtxt(sys.argv[2])
assert len(stats) == len(preds)

# find max preds
pred_arr = np.array(preds)[:,-1]
print(pred_arr[:10])
top = pred_arr.argsort()[-10:][::-1]
#argmax = np.argmax(pred_arr)
#print(argmax, pred_arr[argmax])
print(top, pred_arr[top])
input('enter')

# Iterate over each column and create a seaborn distribution plot
for i in range(stats.shape[1]):
    name = EXTRA_STATS[i]
    top_stats = stats[top, i]
    sns.histplot(stats[:, i], kde=True)
    for ts in top_stats:
        plt.plot([ts,ts],[0,MAX_VALUE],'r-')
    plt.title(f'Distribution of {name}')
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.show()