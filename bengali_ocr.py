# Dependencies:
# pip install opencv-python-headless numpy pytesseract pdf2image
#
# System Dependencies (Required):
# 1. Tesseract OCR with Bengali language data:
#    - Mac: brew install tesseract tesseract-lang
#    - Linux: sudo apt-get install tesseract-ocr tesseract-ocr-ben
# 2. Poppler (for pdf2image):
#    - Mac: brew install poppler
#    - Linux: sudo apt-get install poppler-utils

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import os
import argparse
import sys

def enhance_image_for_ocr(image):
    """
    Preprocesses the image to improve OCR accuracy for scanned Bengali text.
    """
    # Convert PIL Image to numpy array (OpenCV format) if it isn't already
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Denoise (parameters: h=10, templateWindowSize=7, searchWindowSize=21)
    # 'h' regulates filter strength. Higher = cleaner but more detail lost.
    # 10 is a safe starting point for scanned docs.
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This is often better than global thresholding for scanned pages with uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    # Note: We return the processed image. 
    # Tesseract can handle grayscale, but if needed we could binarize here.
    # For now, CLAHE + Denoise is a strong combination for variable quality scans.
    return contrast_enhanced

def extract_text_from_pdf(pdf_path, output_path):
    """
    Converts PDF to images, pre-processes them, and runs Tesseract OCR.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return

    print(f"Processing: {pdf_path}")
    
    try:
        # Convert PDF to list of images
        # dpi=300 is standard for OCR
        pages = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        print(f"Error converting PDF to images. Ensure poppler is installed. Details: {e}")
        return

    extracted_text = []

    print(f"Total pages: {len(pages)}")

    for i, page in enumerate(pages):
        page_num = i + 1
        print(f"Preprocessing and OCR processing page {page_num}...")

        # Preprocess
        processed_image = enhance_image_for_ocr(page)

        # Run OCR
        # lang='ben+eng' looks for Bengali first, then English
        text = pytesseract.image_to_string(processed_image, lang='ben+eng')
        
        # Format output for this page
        page_content = f"--- PAGE {page_num} ---\n\n{text}\n\n"
        extracted_text.append(page_content)

    # Save to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(extracted_text)
        print(f"\nSuccess! Extracted content saved to: {output_path}")
    except IOError as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Extract Bengali/English text from scanned PDF.")
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    
    # Optional: allow user to specify output filename
    parser.add_argument("--output", default="extracted_content.txt", help="Path to the output text file")

    args = parser.parse_args()

    extract_text_from_pdf(args.pdf_path, args.output)
