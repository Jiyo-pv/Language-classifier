import streamlit as st
import joblib
import os
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import cv2
import numpy as np
import platform

# Windows only
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

MODEL_PATH = "model.joblib"

FLAGS = {
    "English":"🇬🇧","French":"🇫🇷","Spanish":"🇪🇸",
    "Portuguese":"🇵🇹","Portugeese":"🇵🇹",
    "Italian":"🇮🇹","German":"🇩🇪","Dutch":"🇳🇱",
    "Russian":"🇷🇺","Greek":"🇬🇷","Arabic":"🇸🇦",
    "Turkish":"🇹🇷","Swedish":"🇸🇪","Danish":"🇩🇰",
    "Hindi":"🇮🇳","Tamil":"🇮🇳",
    "Malayalam":"🇮🇳","Kannada":"🇮🇳","Sanskrit":"🇮🇳"
}

st.set_page_config(page_title="Language Detector", page_icon="🌐")
st.title("🌐 Language Detector")
st.write("Detect language from typed text or uploaded images.")

if not os.path.exists(MODEL_PATH):
    st.error("model.joblib not found.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

def preprocess(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Remove noise
    gray = cv2.medianBlur(gray, 3)

    # Adaptive threshold
    th = cv2.adaptiveThreshold(
        gray,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,11
    )

    return Image.fromarray(th)

def ocr(image):
    processed = preprocess(image)

    langs = (
        "eng+fra+spa+por+ita+deu+nld+rus+ell+ara+tur+"
        "swe+dan+hin+tam+mal+kan+san"
    )

    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 3",
        "--oem 3 --psm 11"
    ]

    best = ""

    for cfg in configs:
        try:
            txt = pytesseract.image_to_string(
                processed,
                lang=langs,
                config=cfg
            )
            if len(txt.strip()) > len(best):
                best = txt
        except:
            pass

    return best

def predict(text):
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    classes = model.classes_

    st.markdown(f"## {FLAGS.get(pred,'🏳️')} {pred}")
    st.progress(int(max(proba)*100))
    st.write(f"**Confidence:** {max(proba)*100:.1f}%")

    df = pd.DataFrame({
        "Language": classes,
        "Probability": proba*100
    }).sort_values("Probability", ascending=False).head(5)

    st.bar_chart(df.set_index("Language"))
    st.dataframe(df.round(1), hide_index=True)

tab1, tab2 = st.tabs(["📝 Text","🖼️ Image"])

with tab1:
    txt = st.text_area("Enter text", height=150)

    if st.button("Detect Language", key="txt"):
        if txt.strip():
            predict(txt)
        else:
            st.warning("Enter some text.")

with tab2:
    file = st.file_uploader(
        "Upload image",
        type=["png","jpg","jpeg","bmp","webp"]
    )

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, use_container_width=True)

        if st.button("Extract Text & Detect", key="img"):
            with st.spinner("Reading text..."):
                text = ocr(image)

            if text.strip():
                st.subheader("Extracted Text")
                st.text_area("", text, height=150)
                predict(text)
            else:
                st.warning("No text detected.")

st.divider()
st.caption("OCR: Tesseract • Language Detection: TF-IDF + Multinomial Naive Bayes")
