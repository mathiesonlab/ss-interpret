"""
Train discriminators with different fc_size values to classify real/simulated data.
Uses the discriminator architecture from pg-gan-zip and logs to wandb.
"""

import argparse
import numpy as np
import keras
import wandb
import os
from sklearn.model_selection import train_test_split

# Add pg-gan-zip to path to import discriminator
from pg_gan.discriminator import TwoPopModel

# Import dataset loading functions
from dataset import load_data, DataGenerator
from utils import POP1_n, POP2_n, PREFIX

class WandbMetricsLogger(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # The `logs` dictionary contains the metrics for the epoch.
        # We can directly log this dictionary to W&B.
        # The 'epoch' is automatically logged by W&B as the step.
        wandb.log(logs)


def create_discriminator(fc_size, learning_rate) -> TwoPopModel:
    """Create a discriminator model with specified fc_size."""
    model = TwoPopModel(POP1_n, POP2_n, fc_size=fc_size)
    
    # Compile the model with binary crossentropy loss and custom metrics
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.Precision(thresholds=0, name='precision'),
            keras.metrics.Recall(thresholds=0, name='recall'),
        ]
    )
    
    return model

def train_discriminator(fc_size, samples, labels, num_epochs=10, batch_size=64, learning_rate=0.001, seed=0, 
                        randomize_labels=False, randomize_weights=False, name=None):
    """Train a discriminator with specified fc_size."""
    name = "disc" if name is None else name
    model_path = PREFIX + f"discs/{name}_{seed}.keras"
    if randomize_weights:
        model_path = PREFIX + f"discs/random-weights_{seed}.keras"
    
    # Set random seeds
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Experiment: shuffle or not
    if randomize_labels:
        rng.shuffle(labels)
    
    # Initialize wandb run
    run = wandb.init(project="ss-interpret", config={
        "fc_size": fc_size,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "exp_randomize": randomize_labels,
        "model_path": model_path
    })

    # Split data
    indices = np.arange(len(samples))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.1, random_state=seed, stratify=labels
    )
    
    print(f"Training discriminator with fc_size={fc_size}")
    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    
    # Create datasets
    train_dataset = DataGenerator(samples, labels, train_indices, batch_size=batch_size, seed=seed)
    test_dataset = DataGenerator(samples, labels, test_indices, batch_size=batch_size, seed=seed)
    
    # Create model
    model = create_discriminator(fc_size, learning_rate)
    
    # Build the model by calling it once
    _ = model(samples[train_indices[:1]], training=False)
    model.summary()

    # save and quit if randomizing weights (initialized)
    if randomize_weights:
        # Final evaluation
        final_results = model.evaluate(test_dataset, verbose=0, return_dict=True)
        final_accuracy = final_results['binary_accuracy']
        
        print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        wandb.finish()
        return model, final_accuracy

    # Train the model
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=test_dataset,
        callbacks=[ 
            WandbMetricsLogger(),
            # keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, start_from_epoch=10)
        ],
        verbose=1
    )
    
    # Final evaluation
    final_results = model.evaluate(test_dataset, verbose=0, return_dict=True)
    final_accuracy = final_results['binary_accuracy']
    
    print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    wandb.finish()
    return model, final_accuracy

def main():
    """Main training function."""
    # take parameters in as arguments
    parser = argparse.ArgumentParser(description="Train discriminator with different fc_size values.")
    #parser.add_argument('--data_path', type=str, help='Path to X.npy, y.npy data')
    parser.add_argument('--fc_size', type=int, default=128, help='Fully connected layer size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--randomize_labels', action="store_true", help='Experiment: randomize labels and see if correlations remain')
    parser.add_argument('--randomize_weights', action="store_true", help='Experiment: randomize weights and see if correlations remain')
    parser.add_argument('--name', type=str, default=None, help='Name for the experiment run')

    args = parser.parse_args()
    # Load data
    print("Loading dataset...")
    samples, labels = load_data(dir="/tmp")

    print(f"Dataset shape: {samples.shape}")
    print(f"Labels distribution: Real={np.sum(labels)}, Simulated={len(labels) - np.sum(labels)}")

    _, _ = train_discriminator(
        fc_size=args.fc_size,
        samples=samples,
        labels=labels,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        randomize_labels=args.randomize_labels,
        randomize_weights=args.randomize_weights,
        name=args.name
    )
     
    
if __name__ == "__main__":
    main()
