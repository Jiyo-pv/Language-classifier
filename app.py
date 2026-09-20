import streamlit as st
import joblib
import os
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import platform
import pytesseract

# Windows Tesseract path
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

MODEL_PATH = "model.joblib"

# Flag emojis
FLAGS = {
    "English": "🇬🇧",
    "French": "🇫🇷",
    "Spanish": "🇪🇸",
    "Portugeese": "🇵🇹",
    "Portuguese": "🇵🇹",
    "Italian": "🇮🇹",
    "German": "🇩🇪",
    "Dutch": "🇳🇱",
    "Russian": "🇷🇺",
    "Greek": "🇬🇷",
    "Arabic": "🇸🇦",
    "Turkish": "🇹🇷",
    "Swedish": "🇸🇪",
    "Danish": "🇩🇰",
    "Hindi": "🇮🇳",
    "Tamil": "🇮🇳",
    "Malayalam": "🇮🇳",
    "Kannada": "🇮🇳",
    "Sanskrit": "🇮🇳",
}

st.set_page_config(
    page_title="Language Detector",
    page_icon="🌐",
    layout="centered"
)

st.title("🌐 Language Detector")
st.write("Detect language from typed text or an uploaded image.")

if not os.path.exists(MODEL_PATH):
    st.error("model.joblib not found. Train the model first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

def show_prediction(text):
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    classes = model.classes_

    confidence = max(proba) * 100

    st.markdown(f"## {FLAGS.get(pred,'🏳️')} {pred}")
    st.progress(int(confidence))
    st.write(f"**Confidence:** {confidence:.1f}%")

    top = (
        pd.DataFrame({
            "Language": classes,
            "Probability": proba * 100
        })
        .sort_values("Probability", ascending=False)
        .head(5)
    )

    st.subheader("Top Predictions")
    st.bar_chart(top.set_index("Language"))
    st.dataframe(
        top.round(1),
        hide_index=True,
        use_container_width=True
    )

tab1, tab2 = st.tabs(["📝 Text", "🖼️ Image"])

# ---------------- TEXT ----------------
with tab1:
    txt = st.text_area(
        "Enter text",
        height=150,
        placeholder="Type any sentence..."
    )

    if st.button("Detect Language", key="txt", type="primary"):
        if txt.strip():
            show_prediction(txt)
        else:
            st.warning("Please enter some text.")

# ---------------- IMAGE ----------------
with tab2:

    file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg", "bmp", "webp"]
    )

    if file:
        image = Image.open(file)
        st.image(image, use_container_width=True)

        if st.button(
            "Extract Text & Detect",
            key="img",
            type="primary"
        ):

            with st.spinner("Running OCR..."):

                # -------- Image preprocessing --------
                gray = ImageOps.grayscale(image)
                gray = gray.filter(ImageFilter.SHARPEN)

                processed = gray.point(
                    lambda x: 0 if x < 160 else 255,
                    "1"
                )

                # 17 language OCR
                LANGS = (
                    "eng+fra+spa+por+ita+deu+nld+rus+"
                    "ell+ara+tur+swe+dan+hin+tam+mal+kan"
                )

                CONFIG = "--oem 3 --psm 6"

                try:
                    extracted = pytesseract.image_to_string(
                        processed,
                        lang=LANGS,
                        config=CONFIG
                    )
                except pytesseract.TesseractError:
                    extracted = pytesseract.image_to_string(
                        processed,
                        lang="eng",
                        config=CONFIG
                    )

            if extracted.strip():
                st.subheader("Extracted Text")
                st.text_area(
                    "",
                    extracted,
                    height=150,
                    disabled=True
                )
                show_prediction(extracted)
            else:
                st.warning("No readable text detected.")

st.divider()
st.caption(
    "TF-IDF Character N-gram + Multinomial Naive Bayes | OCR: Tesseract"
)
