import tensorflow as tf
import numpy as np
import time

# Load the pre-trained model
model = tf.keras.models.load_model('mnist-model.h5')

# Re-compile (optional if already trained and just testing)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load and preprocess MNIST test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}\n")

# Pick one test image to analyze per-layer timing
sample_idx = 0
sample_image = x_test[sample_idx].reshape(1, 28, 28, 1)

# Run and time each layer manually
print("🕒 Timing each layer during inference...\n")
current_output = sample_image

for layer in model.layers:
    t0 = time.time()
    current_output = layer(current_output)
    t1 = time.time()
    
    output_shape = tuple(current_output.shape)
    print(f"{layer.name:<20} | Output shape: {output_shape} | Time: {(t1 - t0)*1000:.3f} ms")

# Final prediction
pred_probs = current_output.numpy()
pred_digit = np.argmax(pred_probs)
print(f"\nPredicted digit for test sample {sample_idx}: {pred_digit}")
print(f"Actual digit: {y_test[sample_idx]}")
