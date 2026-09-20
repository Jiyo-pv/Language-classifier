import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
from PIL import Image
import easyocr

MODEL_PATH = "model.joblib"

FLAGS = {
    "English":"🇬🇧","French":"🇫🇷","Spanish":"🇪🇸","Portugeese":"🇵🇹",
    "Portuguese":"🇵🇹","Italian":"🇮🇹","German":"🇩🇪","Dutch":"🇳🇱",
    "Russian":"🇷🇺","Greek":"🇬🇷","Arabic":"🇸🇦","Turkish":"🇹🇷",
    "Swedish":"🇸🇪","Danish":"🇩🇰","Hindi":"🇮🇳","Tamil":"🇮🇳",
    "Malayalam":"🇮🇳","Kannada":"🇮🇳"
}

st.set_page_config(page_title="Language Detector", page_icon="🌐")

st.title("🌐 Language Detector")
st.write("Detect language from text or an uploaded image.")

if not os.path.exists(MODEL_PATH):
    st.error("model.joblib not found.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_reader():
    return easyocr.Reader(
        ['en','fr','es','pt','it','de','nl','ru',
         'el','ar','tr','sv','da','hi','ta','ml','kn'],
        gpu=False
    )

model = load_model()
reader = load_reader()

def show_prediction(text):
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    classes = model.classes_

    confidence = max(proba) * 100

    st.markdown(f"## {FLAGS.get(pred,'🏳️')} {pred}")
    st.progress(int(confidence))
    st.write(f"**Confidence:** {confidence:.1f}%")

    df = pd.DataFrame({
        "Language": classes,
        "Probability": proba * 100
    }).sort_values("Probability", ascending=False).head(5)

    st.subheader("Top 5 Predictions")
    st.bar_chart(df.set_index("Language"))
    st.dataframe(df.round(1), hide_index=True, use_container_width=True)

tab1, tab2 = st.tabs(["📝 Text", "🖼️ Image"])

with tab1:
    txt = st.text_area("Enter text", height=150)

    if st.button("Detect Language", type="primary"):
        if txt.strip():
            show_prediction(txt)
        else:
            st.warning("Enter some text.")

with tab2:
    file = st.file_uploader(
        "Upload Image",
        type=["png","jpg","jpeg","bmp","webp"]
    )

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, use_container_width=True)

        if st.button("Extract Text & Detect", type="primary"):
            with st.spinner("Reading text..."):
                results = reader.readtext(np.array(image), detail=0)
                extracted = " ".join(results)

            if extracted.strip():
                st.subheader("Extracted Text")
                st.text_area("", extracted, height=150)
                show_prediction(extracted)
            else:
                st.warning("No text detected.")

st.divider()
st.caption("OCR: EasyOCR | Language Detection: TF-IDF + Naive Bayes")
