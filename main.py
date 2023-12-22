import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
import re
import dateutil.parser
import requests
import os
from tensorflow.keras.models import load_model
from pytesseract import Output

from flask import Flask, request, jsonify


def plot_gray(image):
    plt.figure(figsize=(16, 10))
    return plt.imshow(image, cmap='Greys_r')


def plot_rgb(image):
    plt.figure(figsize=(16, 10))
    return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def classify_text(text, max_amount, model_path):
    # Load the pre-trained classification model
    model = load_model(model_path)

    # Check for specific keywords or patterns indicative of different classes
    if re.search(r'\d+\.\d{2}\b', text):
        if float(re.search(r'\d+\.\d{2}\b', text).group()) == max_amount:
            return 'Total'
    elif any(keyword in text.lower() for keyword in ['date', 'time', 'day', 'month', 'year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'am', 'pm']):
        return 'Tanggal'
    elif re.search(r'\b\d{1,4}[/-]\d{1,2}[/-]\d{2,4}\b', text):
        return 'Tanggal'
    elif any(keyword in text.lower() for keyword in ['toko', 'shop', 'store']):
        return 'Nama Toko'
    else:
        # Use the pre-trained model to classify 'Item' category
        text_sequence = [" ".join(text.split())]  # Preprocess text if needed
        item_class = model.predict(text_sequence)[0]
        return item_class


def extract_store_name(text):
    lines = text.split('\n')
    if lines:
        return lines[0].strip()
    return None


def extract_date_and_time(text):
    # Check for specific keywords or patterns indicative of a date
    date_keywords = ['date', 'day', 'month', 'year', 'jan', 'feb', 'mar',
                     'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    # Convert text to lowercase for case-insensitive matching
    lowercase_text = text.lower()

    # Extract only dates
    dates_found = [dateutil.parser.parse(match).date() for match in re.findall(
        r'\b\d{1,4}[/-]\d{1,2}[/-]\d{2,4}\b', text)]

    return dates_found


def find_amounts(text):
    amounts = re.findall(r'\d+\.\d{2}\b', text)
    if not amounts:
        return ['Total tidak ditemukan']

    floats = [float(amount) for amount in amounts]
    unique = list(dict.fromkeys(floats))
    return unique


# file = "Test-Project/1188-receipt.jpg"
model_path = 'model.h5'

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "No file uploaded"})

        try:
            file_content = file.read()
            nparr = np.fromstring(file_content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            d = pytesseract.image_to_data(image, output_type=Output.DICT)
            n_boxes = len(d['level'])
            boxes = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            for i in range(n_boxes):
                (x, y, w, h) = (d['left'][i], d['top']
                                [i], d['width'][i], d['height'][i])
                boxes = cv2.rectangle(
                    boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

            extracted_text = pytesseract.image_to_string(image)

            store_name = extract_store_name(extracted_text)
            # print("Store Name:", store_name)

            amounts = find_amounts(extracted_text)

            if amounts == ['Total tidak ditemukan']:
                max_amount = None
                print("Total tidak ditemukan")
            else:
                max_amount = max(amounts)
                # print("Total:", max_amount)

            dates_found = extract_date_and_time(extracted_text)
            if dates_found:
                # print("Tanggal ditemukan dalam teks:")
                for date_found in dates_found:
                    print(date_found)
            else:
                print("Tanggal tidak ditemukan dalam teks.")

            classification_result = classify_text(
                extracted_text, max_amount, model_path)
            data = {
                "store_name": store_name,
                "total": max_amount,
                "dates_found": [str(date_found) for date_found in dates_found],
            }
            return jsonify(data)

        except Exception as e:
            return jsonify({"error": str(e)})
    return "OK"

# plot_gray(image)
# plot_rgb(boxes)
# print("Extracted Text:")
# print(extracted_text)


if __name__ == "__main__":
<<<<<<< HEAD
    app.run(debug=True)
=======
    app.run(debug=True, port=int(os.environ.get('PORT', 8080)))

>>>>>>> 63056e7fb63eb68cdcc74306da765d3bc40cbab7
