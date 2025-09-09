import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model("mnist-model.h5")

# Print layer names and types
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")

# Extract Conv2D and MaxPooling2D layer weights
for i, layer in enumerate(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        weights, biases = layer.get_weights()
        # Save weights and biases to text files
        np.savetxt(f"conv{i+1}_weights.txt", weights.flatten(), fmt="%.8f")
        np.savetxt(f"conv{i+1}_biases.txt", biases, fmt="%.8f")
        print(f"Saved conv{i+1}_weights.txt and conv{i+1}_biases.txt")

    elif isinstance(layer, tf.keras.layers.MaxPooling2D):
        print(f"Layer {i} is MaxPooling2D - no weights to save.")
