import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('mnist-model.h5')

# Re-compile the model to set optimizer and loss (necessary after loading)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load MNIST test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess test images: normalize pixels to [0,1] and reshape to (28,28,1)
x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# Use integer labels directly (do NOT one-hot encode for sparse_categorical_crossentropy)
# y_test is already integers with shape (num_samples,)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# Predict and print for a sample test image (index 0)
sample_idx = 0
sample_image = x_test[sample_idx].reshape(1, 28, 28, 1)  # Add batch dimension
pred_probs = model.predict(sample_image)
pred_digit = np.argmax(pred_probs)
print(f"Predicted digit for test sample {sample_idx}: {pred_digit}")
print(f"Actual digit: {y_test[sample_idx]}")
