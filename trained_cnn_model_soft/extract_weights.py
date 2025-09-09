import tensorflow as tf

# Load model
model = tf.keras.models.load_model('mnist-model.h5')

# Extract Conv2D layers
conv1 = model.layers[1]  # Conv2D
conv2 = model.layers[2]  # Conv2D

# Get weights and biases
weights1, biases1 = conv1.get_weights()
weights2, biases2 = conv2.get_weights()

# Save conv1 weights
with open("conv1_weights.txt", "w") as f:
    for filt in range(weights1.shape[3]):  # 32 filters
        for i in range(3):
            for j in range(3):
                val = weights1[i][j][0][filt]
                f.write(f"{val:.8f}\n")

# Save conv1 biases
with open("conv1_biases.txt", "w") as f:
    for b in biases1:
        f.write(f"{b:.8f}\n")

# Save conv2 weights
with open("conv2_weights.txt", "w") as f:
    for filt in range(weights2.shape[3]):  # 64 filters
        for c in range(32):  # 32 input channels
            for i in range(3):
                for j in range(3):
                    val = weights2[i][j][c][filt]
                    f.write(f"{val:.8f}\n")

# Save conv2 biases
with open("conv2_biases.txt", "w") as f:
    for b in biases2:
        f.write(f"{b:.8f}\n")

print("✅ Weights and biases saved to .txt files.")
