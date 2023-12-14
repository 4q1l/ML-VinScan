import cv2
import pytesseract
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import re

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Resize the image to match the input size of the model
    image = cv2.resize(image, (224, 224))
    # Reshape the image to match the input shape expected by the model
    image = np.reshape(image, (1, 224, 224, 3))
    # Normalize the pixel values to be in the range [0, 1]
    image = image / 255.0
    return image

def predict_and_extract_text(model_path, image_path):
    # Load the trained model
    model = load_model(model_path)

    # Preprocess the image for prediction
    preprocessed_image = preprocess_image(image_path)

    # Make a prediction using the model
    prediction = model.predict(preprocessed_image)

    # Get class index with the highest probability
    predicted_class_index = np.argmax(prediction)

    # Map the predicted class index to the corresponding class label
    class_labels = ['nama_item', 'harga_item', 'total_harga']
    predicted_class_label = class_labels[predicted_class_index]

    # Use Tesseract OCR to extract text from the image
    image = cv2.imread(image_path)
    extracted_text = pytesseract.image_to_string(image)

    # Perform further processing to extract relevant information from the text
    # You may need to customize this part based on the format of your text

    # Example: Extracting price information
    price_pattern = re.compile(r'\$\s*\d+(\.\d{2})?')
    prices = re.findall(price_pattern, extracted_text)
    if prices:
        price = prices[0]
    else:
        price = "Not found"

    # Example: Extracting total information
    total_pattern = re.compile(r'Total\s*\$\s*\d+(\.\d{2})?')
    totals = re.findall(total_pattern, extracted_text)
    if totals:
        total = totals[0]
    else:
        total = "Not found"

    return {
        'predicted_class_label': predicted_class_label,
        'extracted_text': extracted_text,
        'price': price,
        'total': total
    }

# Example usage
model_path = "model.h5"
image_path = "image.jpg"
result = predict_and_extract_text(model_path, image_path)
print(result)
