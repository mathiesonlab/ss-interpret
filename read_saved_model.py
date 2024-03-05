"""
Obtain the actual weights from a saved model in tensorflow.
Author: Sara Mathieson
Date: 2/18/24
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# our imports
from correlation_heatmap import parse_correlation_file

# this is how to print them all at once, but is not as helpful for getting the actual numbers
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#print_tensors_in_checkpoint_file(file_name=filename,tensor_name="",all_tensors=True)

PATH = "/Users/saramathieson/Dropbox/ss-interpret/"
HIDDEN_PI = "hidden_pi/hidden_pi_CEU_"
SUFFIX = "_230410_230830_finetuneAug23"
NUM_RAND = 10

def main():
    for seed in range(0,5):
        hidden_pi_file = PATH + HIDDEN_PI + str(seed) + SUFFIX + "_GBR.txt"
        all_hidden, common_indices = parse_correlation_file(hidden_pi_file)
        print(common_indices)
        trained_disc = PATH + "trained_discs/CEU/CEU_" + str(seed) + SUFFIX + "/"
        trained_disc += "variables/variables" #.data-00001-of-00002"

        print("seed", seed)
        
        # common indices
        print("useful filters")
        get_weights(trained_disc, common_indices, title="seed " + str(seed) + " useful filters")

        # not common indices
        print("random filters")
        random_indices = []
        while len(random_indices) < NUM_RAND:
            rand = np.random.randint(0, 64)
            if rand not in common_indices and rand not in random_indices:
                random_indices.append(rand)

        get_weights(trained_disc, random_indices, title="seed " + str(seed) + " random filters")

def get_weights(trained_disc, common_indices, title=""):
    # this is the source code from above
    # https://gist.github.com/mvsusp/0eff480bf4848fea05689d8af5394d7c
    #reader = pywrap_tensorflow.NewCheckpointReader(filename)
    reader = tf.train.load_checkpoint(trained_disc)
    var_to_shape_map = reader.get_variable_to_shape_map()
    fig1 = plt.figure()
    plt.title(title)
    num_rows = len(common_indices)
    num_cols = 1
    for key in sorted(var_to_shape_map):
        if key.startswith("conv2/kernel"):
            print("tensor_name: ", key)
            weights = reader.get_tensor(key)
            #print(weights.shape)
            for (i, idx) in enumerate(common_indices):
                filter = weights[0,:,:,idx]
                print("min/max", np.min(filter), np.max(filter))
                fig1 = plt.subplot(num_rows, num_cols, i+1)
                plt.imshow(filter, cmap='gray', vmin=-0.5, vmax=0.5)
                

                plt.tick_params(
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off

                '''plt.tick_params(
                    axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off'''
                
                #plt.ylabel("filter: " + str(idx))
    plt.show()

if __name__ == "__main__":
    main()