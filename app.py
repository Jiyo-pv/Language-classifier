import streamlit as st
import joblib
import os
import pandas as pd

MODEL_PATH = "model.joblib"

FLAGS = {
    "English": "🇬🇧",
    "French": "🇫🇷",
    "Spanish": "🇪🇸",
    "Portuguese": "🇵🇹",
    "Portugeese": "🇵🇹",
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
    "Sanskrit": "🇮🇳"
}

st.set_page_config(
    page_title="Language Detector",
    page_icon="🌐",
    layout="centered"
)

st.title("🌐 Multilingual Language Detector")
st.write(
    "Detect the language of typed or pasted text using a "
    "Linear SVM machine learning model."
)

if not os.path.exists(MODEL_PATH):
    st.error("model.joblib not found. Train the model first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

text = st.text_area(
    "Enter Text",
    height=180,
    placeholder="Type or paste text in any supported language..."
)

if st.button("Detect Language", type="primary"):

    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    confidence = max(probabilities) * 100

    st.markdown(f"## {FLAGS.get(prediction,'🏳️')} {prediction}")
    st.progress(int(confidence))
    st.caption(f"Confidence: {confidence:.1f}%")

    st.subheader("Top 5 Predictions")

    df = (
        pd.DataFrame({
            "Language": classes,
            "Confidence (%)": probabilities * 100
        })
        .sort_values("Confidence (%)", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

    st.bar_chart(df.set_index("Language")["Confidence (%)"])
    st.dataframe(df.round(1), hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "Machine Learning Model: TF-IDF Character N-grams + Linear Support Vector Machine (Linear SVM)"
)
