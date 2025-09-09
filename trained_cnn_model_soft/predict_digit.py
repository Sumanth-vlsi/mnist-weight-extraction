import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load your saved image 'digit_8.png' from the current directory
img_path = 'digit_8.png'
img = Image.open(img_path).convert('L')  # Convert to grayscale

# Resize to 28x28 pixels as expected by the model
img = img.resize((28, 28))

# Convert to numpy array
img_array = np.array(img)

# Optional: visualize the input image to verify
plt.imshow(img_array, cmap='gray')
plt.title("Input image")
plt.axis('off')
plt.show()

# Normalize pixel values to [0,1] (same as training)
img_array = img_array.astype('float32') / 255.0

# Reshape to add batch and channel dimensions: (1, 28, 28, 1)
input_image = img_array.reshape(1, 28, 28, 1)

# Load the pre-trained MNIST model
model = tf.keras.models.load_model('mnist-model.h5')

# Recompile model (necessary after loading)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Predict the digit
pred_probs = model.predict(input_image)
pred_digit = np.argmax(pred_probs)

print(f"Predicted digit for the input image '{img_path}': {pred_digit}")
