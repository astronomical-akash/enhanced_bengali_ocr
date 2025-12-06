# Upgrade command: pip install --upgrade google-generativeai

import streamlit as st
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import google.generativeai as genai
import google.api_core.exceptions
import tempfile
import os
from PIL import Image
import re
import math

# --- Helper Functions ---

def enhance_image_for_ocr(pil_image):
    """
    Preprocesses the image for better OCR results.
    1. Convert to OpenCV format (BGR).
    2. Convert to Grayscale.
    3. Denoise.
    4. Apply CLAHE.
    """
    # Convert PIL to OpenCV format (RGB -> BGR)
    image_np = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Denoise
    # h=10 is a good baseline for scanned documents
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    return contrast_enhanced

def get_valid_model(api_key):
    """
    Dynamically finds a valid Gemini model for the user.
    1. Lists all models available to the API key.
    2. Filters for 'generateContent' support.
    3. Priorities: 'flash' > 'pro' > First Available.
    """
    genai.configure(api_key=api_key)
    try:
        model_list = list(genai.list_models())
        
        # Filter: Must support 'generateContent'
        supported_models = [m for m in model_list if 'generateContent' in m.supported_generation_methods]
        
        if not supported_models:
            raise Exception("No Gemini models found that support 'generateContent'. Check your API key and permissions.")

        # Priority 1: Flash
        for model in supported_models:
            if "flash" in model.name.lower():
                return model.name
        
        # Priority 2: Pro
        for model in supported_models:
            if "pro" in model.name.lower():
                return model.name
                
        # Fallback: Just take the first one
        return supported_models[0].name

    except Exception as e:
        raise Exception(f"Failed to find a valid model: {str(e)}")

def clean_with_gemini(text, api_key):
    """
    Uses Google Gemini to clean and format the OCR output.
    Uses dynamic model selection and batch processing for large texts.
    """
    # 1. Validate Model
    try:
        model_name = get_valid_model(api_key)
    except Exception as e:
        return f"Model Discovery Error: {str(e)}", None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # 2. Split text into pages
        # Regex to split by the delimiter we inserted: --- PAGE {n} ---
        # We capture the content between delimiters.
        # This split might produce empty strings at start/end, so we filter.
        raw_pages = re.split(r'--- PAGE \d+ ---\n', text)
        pages = [p.strip() for p in raw_pages if p.strip()]

        if not pages:
            # Fallback if delimiter check fails (e.g. pasted text without delimiters)
            pages = [text]

        # 3. Create Batches (Chunk size = 5 pages)
        BATCH_SIZE = 5
        batches = [pages[i:i + BATCH_SIZE] for i in range(0, len(pages), BATCH_SIZE)]
        
        cleaned_parts = []
        total_batches = len(batches)
        
        # Streamlit progress bar (requires st to be imported)
        progress_text = "AI Cleaning in progress. Please wait..."
        my_bar = st.progress(0, text=progress_text)

        system_prompt = (
            "You are an expert Bengali and English editor. "
            "Clean OCR noise, correct spellings, and format the output in clean Markdown (headers, bullets). "
            "Keep the syllabus structure exactly as input. "
            "If you encounter jumbled text that looks like a failed OCR of a chart or diagram, "
            "ignore it or summarize it as [Chart/Diagram]. Do not try to translate random noise. "
            "Do not add introductory text like 'Here is the cleaned text'. Just output the cleaned content."
        )

        for idx, batch in enumerate(batches):
            # Join the batch pages back into a string for the context window
            batch_text = "\n\n".join(batch)
            
            full_prompt = f"{system_prompt}\n\nHere is a part of the document (Part {idx+1}/{total_batches}):\n\n{batch_text}"
            
            # Generate
            response = model.generate_content(full_prompt)
            cleaned_parts.append(response.text)
            
            # Update progress
            percent_complete = (idx + 1) / total_batches
            my_bar.progress(percent_complete, text=f"AI Cleaning batch {idx+1}/{total_batches}...")

        my_bar.empty() # Clear progress bar
        
        # 4. Combine
        final_text = "\n\n".join(cleaned_parts)
        return final_text, model_name

    except Exception as e:
        return f"Error connecting to Gemini ({model_name}): {str(e)}", model_name

# --- Main App ---

def main():
    st.set_page_config(page_title="Bengali OCR & AI Cleaner", layout="wide")

    st.title("📄 Bengali OCR & AI Cleaner")
    st.markdown("Upload a scanned PDF to extract Bengali/English text and clean it using AI.")

    # --- Sidebar ---
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    
    active_model_name = None
    if api_key:
        try:
            active_model_name = get_valid_model(api_key)
            st.sidebar.success(f"Connected! Using model:\n`{active_model_name}`")
        except Exception as e:
            st.sidebar.error(f"Connection Error: {e}")

    # --- Session State Management ---
    if 'raw_text' not in st.session_state:
        st.session_state['raw_text'] = ""
    if 'cleaned_text' not in st.session_state:
        st.session_state['cleaned_text'] = ""
    if 'used_model' not in st.session_state:
        st.session_state['used_model'] = ""

    # --- File Upload ---
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

    if uploaded_file is not None:
        # --- Step 1: Extraction ---
        if st.button("Extract Raw Text"):
            with st.spinner("Processing PDF... (Converting -> Preprocessing -> OCR with Layout Analysis)"):
                try:
                    # Save uploaded file to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    images = convert_from_path(tmp_path, dpi=300)
                    os.remove(tmp_path) 

                    extracted_pages = []
                    progress_bar = st.progress(0)
                    total_pages = len(images)

                    for i, page_img in enumerate(images):
                        # Preprocess
                        processed_img = enhance_image_for_ocr(page_img)

                        # Run OCR with PSM 3 (Fully automatic page segmentation)
                        text = pytesseract.image_to_string(processed_img, lang='ben+eng', config='--psm 3')
                        extracted_pages.append(f"--- PAGE {i+1} ---\n{text}\n")

                        progress_bar.progress((i + 1) / total_pages)

                    st.session_state['raw_text'] = "\n".join(extracted_pages)
                    st.success("Extraction Complete!")

                except Exception as e:
                    st.error(f"An error occurred during extraction: {e}")

    # --- Step 2: Display & Edit Raw Text ---
    if st.session_state['raw_text']:
        st.write("### Raw extracted Text")
        edited_text = st.text_area("Edit raw text if needed:", value=st.session_state['raw_text'], height=300)
        
        if edited_text != st.session_state['raw_text']:
            st.session_state['raw_text'] = edited_text

        # --- Step 3: AI Cleaning ---
        if st.button("Clean with Gemini"):
            if not api_key:
                st.error("Please enter a valid Gemini API Key in the sidebar.")
            else:
                current_model_name = active_model_name if active_model_name else "AI"
                with st.spinner(f"Refining using {current_model_name}..."):
                     
                    cleaned_result, model_used = clean_with_gemini(st.session_state['raw_text'], api_key)
                    
                    if cleaned_result and "Error" in cleaned_result and len(cleaned_result) < 200:
                         st.error(cleaned_result)
                    else:
                        st.session_state['cleaned_text'] = cleaned_result
                        st.session_state['used_model'] = model_used
                        st.success(f"AI Cleaning Complete using {model_used}!")

    # --- Step 4: Final Result & Download ---
    if st.session_state['cleaned_text']:
        st.write("---")
        st.write(f"### 🤖 Cleaned & Formatted Text (Model: `{st.session_state.get('used_model', 'Unknown')}`)")
        st.markdown(st.session_state['cleaned_text'])

        st.download_button(
            label="Download Cleaned Text",
            data=st.session_state['cleaned_text'],
            file_name="cleaned_output.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()
