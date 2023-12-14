import cv2
import matplotlib.pyplot as plt
import pytesseract
import re

from pytesseract import Output

# Function to plot grayscale images
def plot_gray(image):
    plt.figure(figsize=(16, 10))
    return plt.imshow(image, cmap='gray')

# Function to plot RGB images
def plot_rgb(image):
    plt.figure(figsize=(16, 10))
    return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Pre-processing part
file_name = "image.jpg"  # Adjust the path accordingly
image = cv2.imread(file_name, cv2.IMREAD_COLOR)

if image is None:
    print(f"Error: Unable to load the image at {file_name}")
else:
    plot_gray(image)

    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    n_boxes = len(d['level'])
    boxes = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plot_rgb(boxes)

    extracted_text = pytesseract.image_to_string(image)
    print(extracted_text)

    def find_amounts(text):
        amounts = re.findall(r'\d+\.\d{2}\b', text)
        floats = [float(amount) for amount in amounts]
        unique = list(dict.fromkeys(floats))
        return unique

    amounts = find_amounts(extracted_text)
    print(amounts)

    if amounts:
        max_amount = max(amounts)
        print(f"Maximum amount: {max_amount}")
    else:
        print("No amounts found.")
