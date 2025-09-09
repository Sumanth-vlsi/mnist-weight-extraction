import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained MNIST model
model = tf.keras.models.load_model('mnist-model.h5')
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary to find the last conv/pooling layer name before flatten 
model.summary()

# Replace with the actual last conv/pooling layer name from your model_summary, e.g. 'max_pooling2d' or 'conv2d_2' etc.
intermediate_layer_name = 'max_pooling2d'  

# Create a model to output activations at this intermediate layer
intermediate_model = tf.keras.Model(inputs=model.input,
                                    outputs=model.get_layer(intermediate_layer_name).output)

# Load and preprocess the test image
img_path = 'digit_8.png'
img = Image.open(img_path).convert('L')  # convert to grayscale
img = img.resize((28, 28))
img_array = np.array(img).astype('float32') / 255.0  # normalize
input_image = img_array.reshape(1, 28, 28, 1)  # add batch and channel dims

# Get intermediate layer output (feature maps)
feature_maps = intermediate_model.predict(input_image)

# Print shape of feature maps
print(f"Feature maps shape: {feature_maps.shape}")

# Print a sample of feature map values say from first channel, top-left 5x5 region
print("Sample feature map values from first filter (channel 0), top-left 5x5 region:")
print(feature_maps[0, 0:5, 0:5, 0])

# Print min, max, mean of feature maps for summary statistics
print(f"Feature maps stats: min={feature_maps.min()}, max={feature_maps.max()}, mean={feature_maps.mean()}")

# Visualize first 16 feature maps (or fewer if less)
num_feature_maps = feature_maps.shape[-1]
num_to_show = min(num_feature_maps, 16)
plt.figure(figsize=(12, 8))
for i in range(num_to_show):
    plt.subplot(4, 4, i+1)
    plt.imshow(feature_maps[0, :, :, i], cmap='gray')
    plt.title(f'Feature map {i}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Predict final digit using the original model
pred_probs = model.predict(input_image)
pred_digit = np.argmax(pred_probs)
print(f"Predicted digit for the input image '{img_path}': {pred_digit}")
