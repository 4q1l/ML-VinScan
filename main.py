import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
import re
import dateutil.parser
import requests
import os
import fitz
from tensorflow.keras.models import load_model
from pytesseract import Output
from werkzeug.utils import secure_filename

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

@app.route("/index2", methods=["POST"])
def index2():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "No file uploaded"})

        try:
            def plot_gray(image):
                plt.figure(figsize=(16, 10))
                return plt.imshow(image, cmap='Greys_r')

            def plot_rgb(image):
                plt.figure(figsize=(16, 10))
                return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            def extract_dates_and_amounts(text):
                # Extract dates in the format dd/mm
                date_pattern = r'\b\d{1,2}/\d{1,2}\b'
                dates = re.findall(date_pattern, text)

                # Extract amounts with IDR and the number on the right
                # Updated pattern for amounts with two decimal places
                amount_pattern1 = r'\b(?:Rp\.|IDR)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:CR|DB)\b'
                amount_pattern2 = r'\b(?:Rp\.|IDR)\s*(\d{1,3}(?:\.\d{3})+(?:,\d{2})?)\s*(?:CR|DB)\b'  # Pattern with dot as decimal separator
                amount_pattern3 = r'\b(?:Rp\.|IDR)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:CR|DB)\b'  # Pattern without thousands separator
                amount_pattern4 = r'\b(?:\d{1,2}/\d{1,2})\s*IDR\s*([\d,]+(?:\.\d{2})?)\b'  # IDR format with date
                amount_pattern5 = r'\bIDR\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b'

                amounts1 = re.findall(amount_pattern1, text)
                amounts2 = re.findall(amount_pattern2, text)
                amounts3 = re.findall(amount_pattern3, text)
                amounts4 = re.findall(amount_pattern4, text)
                amounts5 = re.findall(amount_pattern5, text)

                # Combine amounts from different patterns
                amounts = amounts1 + amounts2 + amounts3 + amounts4 + amounts5

                # Add "Rp." or "IDR" to amounts without it
                amounts = ['Rp. ' + amount if not amount.startswith('Rp.') and not amount.startswith('IDR') else amount for amount in amounts]

                return dates, amounts

            file_content = file.read()
            nparr = np.frombuffer(file_content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            plot_gray(image)

            d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            n_boxes = len(d['level'])
            boxes = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            for i in range(n_boxes):
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

            plot_rgb(boxes)

            extracted_text = pytesseract.image_to_string(image)
    #       print("Extracted Text:")
    #       print(extracted_text)

            # Extract dates and amounts using the new function
            dates, amounts = extract_dates_and_amounts(extracted_text)

            # Print the extracted dates and amounts
            print("Dates:", dates)
            print("Amounts:", amounts)

            for date, amount in zip(dates, amounts):
                print("Date:", date)
                print("Amount:", amount)
                print("-----------")

                # Ubah hasil ke format yang bisa di-return sebagai JSON
            data = {
                "dates": [str(date) for date in dates],
                "amounts": amounts,
            }

            return jsonify(dates=dates, amounts=amounts)

        except Exception as e:
            return jsonify({"error": str(e)})
    return "OK"

@app.route("/index3", methods=["POST"])
def index3():  # Changed the function name
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "No file uploaded"})

        try:
            def plot_gray(image):
                plt.figure(figsize=(16, 10))
                return plt.imshow(image, cmap='Greys_r')

            def plot_rgb(image):
                plt.figure(figsize=(16, 10))
                return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            def extract_dates_and_amounts(text):
                # Initialize a list to store dictionaries of dates and amounts
                data_list = []

                # Split the text into lines
                lines = text.split('\n')

                # Initialize variables to store dates and amounts
                dates = None
                amounts = []

                # Loop through each line
                for line in lines:
                    # Extract dates in the format dd Mmm yyyy
                    date_pattern = r'\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|Mei|Jun|Jul|Ags|Sep|Okt|Nov|Des)\s\d{4}\b'
                    date_matches = re.findall(date_pattern, line)
                    if date_matches:
                        # Update the current date
                        dates = date_matches[0]

                    # Extract amounts with IDR and the number on the right
                    amount_pattern = r'\b(?:\+|\-)?\s*Rp\.?\s*([\d,.]+)\b'
                    amount_matches = re.findall(amount_pattern, line)
                    if amount_matches:
                        # Add "Rp." to amounts without it
                        amounts.extend(
                            ['Rp. ' + amount if not amount.startswith('Rp.') else amount for amount in amount_matches])

                # Create dictionaries for each date and amount combination
                if dates is not None:
                    for amount in amounts:
                        data_dict = {
                            "Dates": dates,
                            "Amounts": amount
                        }
                        data_list.append(data_dict)

                return data_list

            file_content = file.read()
            nparr = np.frombuffer(file_content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            plot_gray(image)

            d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            n_boxes = len(d['level'])
            boxes = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            for i in range(n_boxes):
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

            plot_rgb(boxes)

            extracted_text = pytesseract.image_to_string(image)
            #            print("Extracted Text:")
            #            print(extracted_text)

            # Extract dates and amounts using the new function
            data_list = extract_dates_and_amounts(extracted_text)
            return jsonify(data_list)

        except Exception as e:
            return jsonify({"error": str(e)})
    return "OK"

@app.route("/index4", methods=["POST"])
def index4():  # Changed the function name
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "No file uploaded"})

        try:
            def plot_text(text):
                # Print the extracted text with line breaks
                print("Extracted Text:")
                print(text)
                print("--------------------")

            def extract_dates_and_amounts(text):
                # Extract dates in the format dd/mm
                date_pattern = r'\b\d{2}/\d{2}\b'
                dates = re.findall(date_pattern, text)

                # Extract amounts based on their position in the text
                amount_pattern = r'(?:DEBIT|CR)\s*([\d,]+\.\d{2})\s*DB'
                amounts = re.findall(amount_pattern, text)

                # Add "Rp." or "IDR" to amounts without it
                amounts = ['Rp. ' + amount if not amount.startswith('Rp.') and not amount.startswith('IDR') else amount
                           for amount in amounts]

                return dates, amounts

            def extract_dates_and_amounts_from_pdf(pdf_path):
                dates = []
                amounts = []

                # Open the PDF file
                pdf_document = fitz.open(pdf_path)

                # Loop through pages
                for page_number in range(pdf_document.page_count):
                    # Get the page
                    page = pdf_document[page_number]

                    # Extract text from the page
                    text = page.get_text()

                    # Print the extracted text for each page
                    plot_text(text)

                    # Extract dates and amounts from the text
                    page_dates, page_amounts = extract_dates_and_amounts(text)

                    # Append to the overall lists
                    dates.extend(page_dates)
                    amounts.extend(page_amounts)

                # Close the PDF document
                pdf_document.close()

                return dates, amounts

            original_filename = secure_filename(file.filename)

            # Save the file content to a file with the original name
            pdf_file_path = os.path.join(original_filename)
            file.save(pdf_file_path)

            dates, amounts = extract_dates_and_amounts_from_pdf(pdf_file_path)

            # Print the extracted dates and amounts
            print("Dates:", dates)
            print("Amounts:", amounts)

            # Loop through the extracted dates and amounts
            for date, amount in zip(dates, amounts):
                print("Date:", date)
                print("Amount:", amount)
                print("-----------")

            data = {
                "dates": [str(date) for date in dates],
                "amounts": amounts,
            }

            return jsonify(dates=dates, amounts=amounts)

        except Exception as e:
            return jsonify({"error": str(e)})
    return "OK"

#plot_gray(image)
#plot_rgb(boxes)
#print("Extracted Text:")
#print(extracted_text)

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get('PORT', 8080)))


