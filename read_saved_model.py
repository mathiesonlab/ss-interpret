
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from correlation_heatmap import parse_correlation_file

hidden_pi_file = "/Users/smathieson/Dropbox/ss-interpret/hidden_pi/hidden_pi_CEU_0_230410_finetune_GBR.txt"
all_hidden, common_indices = parse_correlation_file(hidden_pi_file)
print(common_indices)
path = "/Users/smathieson/Dropbox/ss-interpret/trained_discs/CEU/CEU_0_230410_finetune/"
variables = "variables/variables" #.data-00001-of-00002"
filename = path + variables

# this is how to print them all at once
#print_tensors_in_checkpoint_file(file_name=filename,tensor_name="",all_tensors=True)



# this is the source code from above
# https://gist.github.com/mvsusp/0eff480bf4848fea05689d8af5394d7c
#reader = pywrap_tensorflow.NewCheckpointReader(filename)
reader = tf.train.load_checkpoint(filename)
var_to_shape_map = reader.get_variable_to_shape_map()
fig1 = plt.figure()
num_rows = len(common_indices)
num_cols = 1
for key in sorted(var_to_shape_map):
    if key.startswith("conv2/kernel"):
        print("tensor_name: ", key)
        weights = reader.get_tensor(key)
        print(weights.shape)
        for (i, idx) in enumerate(common_indices):
            filter = weights[0,:,:,idx]
            print(filter.shape)
            fig1 = plt.subplot(num_rows, num_cols, i+1)
            plt.imshow(filter, cmap='gray')
plt.show()