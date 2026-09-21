"""
app.py
Streamlit UI for the language detection model.
Supports two input modes: typed/pasted text, or an uploaded image (OCR extracts
the text first, then the same model predicts the language).

Usage:
    streamlit run app.py

Requires Tesseract OCR installed on your system (not just the Python package):
    - Windows: https://github.com/UB-Mannheim/tesseract/wiki
    - Mac:     brew install tesseract
    - Linux:   sudo apt install tesseract-ocr
    - Streamlit Cloud: add a packages.txt file (see README) — no code change needed
"""

import streamlit as st
import joblib
import os
import platform
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
import pytesseract

# Only point to the Windows install location when actually running on Windows.
# On Streamlit Cloud / Linux / Mac, Tesseract installed via packages.txt or
# brew is already on the system PATH, so pytesseract finds it automatically.
if platform.system() == "Windows":
    windows_tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(windows_tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = windows_tesseract_path

MODEL_PATH = "model.joblib"

# Flag emojis for common languages in the dataset (purely cosmetic, extend as needed)
FLAGS = {
    "English": "🇬🇧", "French": "🇫🇷", "Spanish": "🇪🇸", "Portugeese": "🇵🇹",
    "Portuguese": "🇵🇹", "Italian": "🇮🇹", "German": "🇩🇪", "Dutch": "🇳🇱",
    "Russian": "🇷🇺", "Greek": "🇬🇷", "Arabic": "🇸🇦", "Turkish": "🇹🇷",
    "Swedish": "🇸🇪", "Danish": "🇩🇰", "Hindi": "🇮🇳", "Tamil": "🇮🇳",
    "Malayalam": "🇮🇳", "Kannada": "🇮🇳", "Sanskrit": "🇮🇳",
    "Chinese": "🇨🇳", "Japanese": "🇯🇵", "Korean": "🇰🇷",
}

st.set_page_config(page_title="Language Detector", page_icon="🌐", layout="centered")

st.title("🌐 Language Detector")
st.write("Type text or upload an image, and I'll guess what language it's written in.")

if not os.path.exists(MODEL_PATH):
    st.error(
        "No trained model found (model.joblib missing). "
        "Run `python train.py` first to train and save the model."
    )
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Basic preprocessing to improve OCR accuracy, especially on lower-quality
    Tesseract builds (e.g. older versions installed via apt on cloud hosts)."""
    img = ImageOps.grayscale(image)
    if img.width < 1000:
        scale = 1000 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.5)
    return img


def show_prediction(text: str):
    """Runs the model on `text` and renders the result."""
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    classes = model.classes_

    flag = FLAGS.get(pred, "🏳️")
    confidence = max(proba) * 100
    st.markdown(f"## {flag} **{pred}**")
    st.progress(min(int(confidence), 100))
    st.caption(f"Confidence: {confidence:.1f}%")

    st.subheader("Top guesses")
    top_df = (
        pd.DataFrame({"Language": classes, "Probability": proba})
        .sort_values("Probability", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    top_df["Probability"] = (top_df["Probability"] * 100).round(1)
    st.bar_chart(top_df.set_index("Language")["Probability"])
    st.dataframe(top_df, use_container_width=True, hide_index=True)


tab_text, tab_image = st.tabs(["📝 Text", "🖼️ Image"])

with tab_text:
    text = st.text_area("Enter text", height=150, placeholder="e.g. Bonjour, comment allez-vous ?")
    detect_text_clicked = st.button("Detect Language", type="primary", key="detect_text")

    if detect_text_clicked:
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            show_prediction(text)

with tab_image:
    st.write("Upload an image containing text (a screenshot, photo of a sign, a scanned page, etc.).")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        detect_image_clicked = st.button("Extract Text & Detect Language", type="primary", key="detect_image")

        if detect_image_clicked:
            with st.spinner("Running OCR..."):
                try:
                    processed_image = preprocess_for_ocr(image)
                    # Tesseract only uses English by default. Passing multiple
                    # language codes lets it recognize non-Latin scripts too
                    # (Russian, Arabic, Greek, Hindi/Tamil/Malayalam/Kannada).
                    # Each of these must have its .traineddata file installed
                    # (tick them in the Tesseract Windows installer, or
                    # listed in packages.txt for Streamlit Cloud).
                    lang_codes = "eng+rus+ara+ell+hin+tam+mal+kan"
                    try:
                        extracted_text = pytesseract.image_to_string(processed_image, lang=lang_codes)
                    except pytesseract.pytesseract.TesseractError:
                        # Fallback if some of those language packs aren't installed
                        extracted_text = pytesseract.image_to_string(processed_image)
                except pytesseract.TesseractNotFoundError:
                    st.error(
                        "Tesseract OCR is not installed on this system. "
                        "See the instructions at the top of app.py to install it, "
                        "then restart the app."
                    )
                    st.stop()

            if not extracted_text.strip():
                st.warning("No text could be detected in this image. Try a clearer or higher-resolution image.")
            else:
                st.subheader("Extracted text")
                st.text_area("OCR result", value=extracted_text, height=100, disabled=True)
                show_prediction(extracted_text)

st.divider()
st.caption("Model: TF-IDF (character n-grams) + LinearSVC, trained on the Kaggle Language Detection dataset. "
           "Image text extraction via Tesseract OCR.")
try:
    st.caption(f"Tesseract version: {pytesseract.get_tesseract_version()}")
except Exception:
    pass