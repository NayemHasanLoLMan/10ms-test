import os
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Set the path to your Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust the path for your system

# Define a function to preprocess the image for better OCR results
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding for better text visibility
    _, thresh_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)
    return thresh_img

# Define a function to extract text from an image file
def extract_text_from_image(image_path):
    img = preprocess_image(image_path)
    # Use Tesseract to extract text, specify the language as Bangla (ben)
    text = pytesseract.image_to_string(img, lang='ben')
    return text

# Define the folder containing images
image_folder = 'C:\\Users\\hasan\\Downloads\\HSC26-Bangla1st-Paper'  # Update this with the folder path

# List all files in the folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Extract text from each image
extracted_texts = []
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    text = extract_text_from_image(image_path)
    extracted_texts.append(text)

# Combine all extracted texts into one string
combined_text = "\n\n".join(extracted_texts)

# Define the output text file path
output_file_path = 'extracted_text_bangla.txt'  # Update this to where you want to save the .txt file

# Save the extracted text to the .txt file
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(combined_text)

print(f"Text extracted and saved to {output_file_path}")
