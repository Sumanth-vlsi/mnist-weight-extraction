import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST test dataset
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Find indices of digit '8' in the test set
indices_of_eight = np.where(y_test == 8)[0]

# Select the first image of digit '8'
eight_index = indices_of_eight[0]
eight_image = x_test[eight_index]

# Save the image as 'digit_8.png' in the current directory
plt.imsave('digit_8.png', eight_image, cmap='gray')  # 'gray' colormap for grayscale

print(f"Digit '8' image saved as 'digit_8.png' in your current directory.")
