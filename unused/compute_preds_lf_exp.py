"""
Compute predictions and learned features for the dataset generated in dataset.py.
Uses logic from genome_stats.py.

Version for the experiment of cascading randomizing the model layers.
"""

import os
import sys
from dataset import load_data
import tensorflow as tf
from utils import get_model, compute_all_for_dataset, save_preds_lf

def randomize_layer_weights(layer):
    """
    Re-initializes the weights of a Keras layer using its default initializers.

    Parameters:
    -----------
    layer : tf.keras.layers.Layer
        The layer whose weights are to be randomized.
    """
    if hasattr(layer, 'kernel_initializer'):
        # Get the layer's initializer
        initializer = layer.kernel_initializer
        # Get the shape of the kernel (weights)
        kernel_shape = layer.kernel.shape
        # Generate new random weights using the initializer
        new_kernel = initializer(shape=kernel_shape)
        new_weights = [new_kernel]

        if layer.use_bias:
            # Get the bias initializer and shape
            bias_initializer = layer.bias_initializer
            bias_shape = layer.bias.shape
            # Generate new random biases
            new_bias = bias_initializer(shape=bias_shape)
            new_weights.append(new_bias)

        # Set the new weights
        layer.set_weights(new_weights)
        print(f"    -> Weights for layer '{layer.name}' have been randomized.")
    else:
        print(f"    -> Layer '{layer.name}' does not have weights to randomize.")


def main(pop, model_path, fc_size, max_samples=None):
    # Load the dataset
    print("Loading dataset...")
    samples, _ = load_data(pop)
    print(f"Dataset shape: {samples.shape}")
    
    if max_samples is not None:
        print(f"Processing only {max_samples} samples")

    print(f"\n--- Starting Cascading Randomization for model: {model_path} ---")

    # Load the base model architecture
    model = get_model(model_path, samples[0:1], fc_size=fc_size)
    model_name_base = os.path.basename(model_path).split(".")[0]
    
    # trainable layers only
    trainable_layers = [l for l in model.layers if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.Dense))]
    layers_to_randomize = list(reversed(trainable_layers))
    print(f"\nFound {len(layers_to_randomize)} trainable layers to randomize: {[l.name for l in layers_to_randomize]}")

    for i, layer_to_rand in enumerate(layers_to_randomize):
        step_num = i + 1
        print(f"\n[Step {step_num}/{len(layers_to_randomize)}] Randomizing layer: '{layer_to_rand.name}'")
        randomize_layer_weights(layer_to_rand)
        
        randomized_model_name = f"{model_name_base}_rand_{layer_to_rand.name}"
        
        # Compute predictions and learned features for the partially randomized model
        preds, lf = compute_all_for_dataset(model, samples, max_samples=max_samples, benchmark=False)

        # Save results for this randomization step
        save_preds_lf(pop, randomized_model_name, preds, lf)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nComputes preds and lf for a progressively randomized model, starting with <model_path>.\n")
        print("Usage: python compute_preds_lf_exp.py <population> <model_path> [fc_size]")
        print("Example: python compute_preds_lf_exp.py CEU discs/disc_0.keras 64")
        sys.exit(1)

    pop = sys.argv[1]
    model_path = sys.argv[2]
    fc_size = 64 if len(sys.argv) < 4 else sys.argv[3]

    main(pop, model_path, fc_size, max_samples=None)
