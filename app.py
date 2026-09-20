import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

MODEL_PATH = "model.joblib"

FLAGS = {
    "English":"🇬🇧","French":"🇫🇷","Spanish":"🇪🇸","Portugeese":"🇵🇹",
    "Portuguese":"🇵🇹","Italian":"🇮🇹","German":"🇩🇪","Dutch":"🇳🇱",
    "Russian":"🇷🇺","Greek":"🇬🇷","Arabic":"🇸🇦","Turkish":"🇹🇷",
    "Swedish":"🇸🇪","Danish":"🇩🇰","Hindi":"🇮🇳","Tamil":"🇮🇳",
    "Malayalam":"🇮🇳","Kannada":"🇮🇳","Sanskrit":"🇮🇳"
}

st.set_page_config(page_title="Language Detector", page_icon="🌐")
st.title("🌐 Language Detector")
st.write("Detect language from text or image with automatic OCR.")

if not os.path.exists(MODEL_PATH):
    st.error("model.joblib not found.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# OCR readers (loaded only once)
@st.cache_resource
def ocr_latin():
    return PaddleOCR(lang="en")

@st.cache_resource
def ocr_malayalam():
    return PaddleOCR(lang="ml")

@st.cache_resource
def ocr_tamil():
    return PaddleOCR(lang="ta")

@st.cache_resource
def ocr_kannada():
    return PaddleOCR(lang="kannada")

@st.cache_resource
def ocr_hindi():
    return PaddleOCR(lang="hi")

@st.cache_resource
def ocr_arabic():
    return PaddleOCR(lang="ar")

@st.cache_resource
def ocr_cyrillic():
    return PaddleOCR(lang="ru")

model = load_model()

def extract(reader, img):
    result = reader.predict(np.array(img))
    if result and len(result):
        return " ".join(result[0].get("rec_texts", []))
    return ""

def auto_ocr(img):
    readers = [
        ocr_latin(),
        ocr_malayalam(),
        ocr_tamil(),
        ocr_kannada(),
        ocr_hindi(),
        ocr_arabic(),
        ocr_cyrillic()
    ]

    best = ""
    for r in readers:
        txt = extract(r, img)
        if len(txt.strip()) > len(best):
            best = txt
    return best

def show_prediction(text):
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
    st.dataframe(df.round(1), hide_index=True, use_container_width=True)

tab1, tab2 = st.tabs(["📝 Text", "🖼️ Image"])

with tab1:
    txt = st.text_area("Enter text", height=150)
    if st.button("Detect Language"):
        if txt.strip():
            show_prediction(txt)

with tab2:
    file = st.file_uploader("Upload image", type=["png","jpg","jpeg","webp","bmp"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, use_container_width=True)

        if st.button("Extract & Detect"):
            with st.spinner("Reading text..."):
                text = auto_ocr(img)

            if text.strip():
                st.subheader("Extracted Text")
                st.text_area("", text, height=120)
                show_prediction(text)
            else:
                st.warning("No text detected.")
