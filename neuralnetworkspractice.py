import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np  

# Load MNIST dataset (Chapter 1: Understanding real-world applications of neural networks)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to [0, 1] range (Chapter 2: Preprocessing for efficient model training)
train_images = train_images / 255.0
test_images = test_images / 255.0

# One-hot encode labels (Chapter 2: Data representation in neural networks)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Define the neural network architecture (Chapter 2: Understanding layers and activation functions)
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images to 1D array of 784 pixels
    Dense(128, activation='relu'),  # A hidden layer with 128 neurons and ReLU activation (Chapter 2: Activation functions)
    Dense(10, activation='softmax') # Output layer with 10 neurons (one for each digit) and softmax activation
])

# Compile the model (Chapter 2: Model compilation and understanding optimization)
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Fit the model on training data (Chapter 2: The training process and backpropagation)
model.fit(train_images, train_labels, epochs=5, validation_split=0.2)

# Evaluate the model on test data (Chapter 2: Evaluating model performance)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc}")

# Make predictions (Chapter 1: Neural network's predictive capabilities)
predictions = model.predict(test_images)

# Function to plot images and predictions (Chapter 1: Visualizing neural network predictions)
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = tf.argmax(predictions_array)
    if predicted_label == tf.argmax(true_label):
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f"{predicted_label.numpy()} ({100*np.max(predictions_array):2.0f}%)", color=color)

# Plot the first X test images with their predicted labels (Chapter 1: Interpretation of results)
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
plt.tight_layout()
plt.show()

