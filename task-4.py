# ==========================================
# HANDWRITTEN DIGIT RECOGNITION USING CNN
# ==========================================

# Import Libraries
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# LOAD MNIST DATASET
# ==========================================

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("Dataset Loaded Successfully")


# ==========================================
# NORMALIZE DATA
# ==========================================

x_train = x_train / 255.0
x_test = x_test / 255.0


# ==========================================
# BUILD NEURAL NETWORK MODEL
# ==========================================

model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])


# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(
    x_train,
    y_train,
    epochs=5
)

print("\nModel Training Completed")


# ==========================================
# EVALUATE MODEL
# ==========================================

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", round(test_accuracy * 100, 2), "%")


# ==========================================
# PREDICT DIGITS
# ==========================================

predictions = model.predict(x_test)


# ==========================================
# SHOW SAMPLE PREDICTIONS
# ==========================================

for i in range(5):

    plt.figure(figsize=(3,3))

    plt.imshow(x_test[i], cmap='gray')

    plt.title(
        f"Predicted: {np.argmax(predictions[i])}"
    )

    plt.axis('off')

    plt.savefig(f"digit_prediction_{i}.png")

    plt.close()

    print(f"Saved Image: digit_prediction_{i}.png")
