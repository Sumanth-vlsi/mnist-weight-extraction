import tensorflow as tf
import numpy as np
import cv2

# === Load the trained Sequential model ===
model = tf.keras.models.load_model("mnist-model.h5")

# === Load and preprocess the input image ===
img = cv2.imread("digit_8.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img.astype(np.float32) / 255.0
input_image = img.reshape((1, 28, 28, 1))  # shape: (1, 28, 28, 1)

# Optional: Save input for debugging
np.savetxt("input_image.txt", img, fmt="%.6f")

# === Build a new model using Functional API ===
input_tensor = tf.keras.Input(shape=(28, 28, 1))
x = input_tensor
outputs = []
layer_names = []

for layer in model.layers:
    x = layer(x)
    outputs.append(x)
    layer_names.append(layer.name)

# Create model to output all intermediate layers
intermediate_model = tf.keras.Model(inputs=input_tensor, outputs=outputs)

# === Run inference ===
layer_outputs = intermediate_model.predict(input_image)

# === Save each layer output ===
for i, (layer_name, output) in enumerate(zip(layer_names, layer_outputs)):
    squeezed = output.squeeze()                  # remove batch dimension
    flattened = squeezed.flatten()               # flatten to 1D
    filename = f"layer{i+1}_{layer_name}_output.txt"
    np.savetxt(filename, flattened, fmt="%.6f")  # save to .txt
    print(f"✅ Saved: {filename} | shape: {squeezed.shape} | total values: {flattened.size}")
