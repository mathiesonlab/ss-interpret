import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from pg_gan.discriminator import OnePopModel

def trace(model, input):
    x = input
    for layer in [model.conv1, model.pool, 
                  model.conv2, model.pool, 
                  model.reduce, 
                  model.flatten, 
                  model.fc1, model.dropout, 
                  model.fc2, model.dropout, 
                  model.dense3]:
        x = layer(x)
        x_np = x.numpy()
        print(f"{layer.name:15} | mean: {x_np.mean():8.3f} | std: {x_np.std():8.3f} | min: {x_np.min():8.3f} | max: {x_np.max():8.3f}")

        if layer.name == "reduce_sum" or layer.name == "fc2":
            # plot distribution of outputs
            plt.hist(x_np.flatten(), bins=50)
            plt.title(f"Distribution of {layer.name} outputs")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.savefig(f"unused/{layer.name}_distribution.png")
            plt.close()

        if layer.name == "fc2":
            # get indices of non-zero mean output neurons
            non_zero_indices = tf.where(x_np.mean(axis=0) > 0).numpy()

            # get mean, sd of matching weights in the next layer
            matching_weights = model.dense3.weights[0].numpy()[non_zero_indices[:, 0]]
            print(f"Non-zero weights: {matching_weights.mean()} ({matching_weights.std()}) | {matching_weights.min()} - {matching_weights.max()}")

    out = tf.math.sigmoid(x)
    print(f"{'softmax':15} | mean: {out.numpy().mean():8.3f} | std: {out.numpy().std():8.3f} | min: {out.numpy().min():8.3f} | max: {out.numpy().max():8.3f}")

smp = np.load("dataset-CEU/X.npy")
model = OnePopModel(fc_size=64)
_ = model(smp[:1], training=False)

# print dense3 weights
dense3_weights = model.dense3.weights[0].numpy()
print(f"{'LL WEIGHT':15} | mean: {dense3_weights.mean():8.3f} | std: {dense3_weights.std():8.3f} | min: {dense3_weights.min():8.3f} | max: {dense3_weights.max():8.3f}")
print("-" * 80)
trace(model, smp[:100])

model.load_weights("discs/CEU/disc_0.keras")
print(f"{'LL WEIGHT':15} | mean: {dense3_weights.mean():8.3f} | std: {dense3_weights.std():8.3f} | min: {dense3_weights.min():8.3f} | max: {dense3_weights.max():8.3f}")
print("-" * 80)
trace(model, smp[:100])