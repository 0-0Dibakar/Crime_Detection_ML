import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load the pre-trained ResNet-50 model (you can use other models as well)
model = keras.applications.ResNet50(weights='imagenet')

# Define a function to classify an image
def classify_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Predict the class probabilities
    preds = model.predict(img)

    # Decode and return the top predicted class
    decoded_preds = decode_predictions(preds, top=1)[0]
    return decoded_preds[0]

# Example usage
image_path = 'path_to_crime_activity_image.jpg'  # Replace with the path to your image
predicted_class = classify_image(image_path)
print(f'Predicted class: {predicted_class[1]} with probability: {predicted_class[2]:.2f}')
