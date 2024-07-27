# Import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

# Create the validation data generator
val_images = validation_datagen.flow_from_directory(
    'datasets/animals10/raw-img/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    seed=42
)

test_images, test_labels = next(val_images)

# Load the best saved model
best_model_path = 'models/checkpoints/model-01-0.06.keras'
model = tf.keras.models.load_model(best_model_path)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test Loss: {test_loss:.5f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Make predictions
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Visualization of some predictions
def plot_predictions(images, true_labels, predicted_labels):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"True: {true_labels[i]}, Pred: {predicted_labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_predictions(test_images, true_classes, predicted_classes)
