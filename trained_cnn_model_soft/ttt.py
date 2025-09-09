import numpy as np

# ===== File paths =====
image_path = "/home/sumanth/Music/new/layer1_reshape_output.txt"
weights_path = "/home/sumanth/Music/new/conv1_weights.txt"
biases_path = "/home/sumanth/Music/new/conv1_biases.txt"
output_path = "/home/sumanth/Music/new/conv1_output_relu.txt"

# ===== Load data =====
image_pixels = np.loadtxt(image_path, dtype=np.float32)
weights = np.loadtxt(weights_path, dtype=np.float32)
biases = np.loadtxt(biases_path, dtype=np.float32)

# ===== Reshape image =====
img_2d = image_pixels.reshape(28, 28)

# ===== Prepare output =====
output_feature_map = np.zeros((26, 26, 32), dtype=np.float32)

# ===== Convolution loop =====
for f in range(32):  # For each filter
    kernel = weights[f*9:(f+1)*9].reshape(3, 3)
    bias = biases[f]
    for y in range(26):  # Output height
        for x in range(26):  # Output width
            patch = img_2d[y:y+3, x:x+3]
            conv_sum = np.sum(patch * kernel) + bias
            relu_sum = max(conv_sum, 0)
            output_feature_map[y, x, f] = relu_sum

# ===== Save output =====
# Flatten as in C-style row-major order: filter 0 all values, then filter 1, ...
with open(output_path, "w") as f:
    for f_idx in range(32):
        for y in range(26):
            for x in range(26):
                f.write(f"{output_feature_map[y, x, f_idx]:.6f}\n")

print(f"✅ Convolution + ReLU done. Output saved to {output_path}")
