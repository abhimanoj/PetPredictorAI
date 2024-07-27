from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load class names (assuming you have this list from your training data)
class_names = [
    "cane",
    "cavallo",
    "elefante",
    "farfalla",
    "gallina",
    "gatto",
    "mucca",
    "pecora",
    "scoiattolo",
    "ragno",
]

# Mapping from class names to English translations
translate = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "ragno": "spider",
}

# Load the image file
img_path = "image.jpeg"
img = image.load_img(img_path, target_size=(224, 224))

# Display the original image
plt.imshow(img)
plt.title("Loaded Image")
plt.show()

# Convert the image to a numpy array
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Create a batch

# Preprocess the image
img_array_preprocessed = preprocess_input(img_array.copy())

# Load the trained model
model_path = "models/checkpoints/model-01-0.06.keras"
model = tf.keras.models.load_model(model_path)

# Predict the class
predictions = model.predict(img_array_preprocessed)
predicted_class_index = np.argmax(predictions, axis=1)[0]
predicted_class_name = class_names[predicted_class_index]
confidence_score = predictions[0][predicted_class_index]

# Translate the predicted class name to English
predicted_class_english = translate[predicted_class_name]

# Print the predicted class and confidence score
print(f"Predicted class (Italian): {predicted_class_name}")
print(f"Predicted class (English): {predicted_class_english}")
print(f"Confidence score: {confidence_score * 100:.2f}%")

# Display the image with the prediction
plt.imshow(img)
title = f"Predicted: {predicted_class_english} ({predicted_class_name})\nConfidence: {confidence_score * 100:.2f}%"
print(title)
plt.title(title)
plt.show()
