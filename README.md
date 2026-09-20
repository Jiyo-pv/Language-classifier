# 🌐 Language Detector

Detects the language of a piece of text using TF-IDF character n-grams + a
Naive Bayes classifier, with a Streamlit UI.

## 1. Get the dataset

Download **"Language Detection.csv"** from Kaggle:
https://www.kaggle.com/datasets/basilb2s/language-detection

Place it here:
```
lang-detect/
├── data/
│   └── Language Detection.csv   <-- put it here
├── train.py
├── app.py
└── ...
```

The dataset has ~10,000 rows across 17 languages (English, French, Spanish,
Portuguese, Italian, German, Dutch, Russian, Greek, Arabic, Turkish, Swedish,
Danish, Hindi, Tamil, Malayalam, Kannada).

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

### Also install Tesseract OCR (for the image upload feature)

The Python package `pytesseract` is just a wrapper — it needs the actual
Tesseract engine installed on your system:

- **Windows**: install from https://github.com/UB-Mannheim/tesseract/wiki,
  then make sure the install folder (e.g. `C:\Program Files\Tesseract-OCR`)
  is on your PATH, or add this line near the top of `app.py`:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```
- **Mac**: `brew install tesseract`
- **Linux (Debian/Ubuntu)**: `sudo apt install tesseract-ocr`

## 3. Train the model

```bash
python train.py
```

This prints accuracy/classification report and saves `model.joblib`.
Typical test accuracy is **95%+**.

## 4. Run the app

```bash
streamlit run app.py
```

Opens a browser UI with two tabs:
- **Text**: paste text directly and get an instant language prediction.
- **Image**: upload a photo/screenshot containing text — it runs OCR
  (Tesseract) to extract the text, shows you what it read, then predicts
  the language.

Both modes show a confidence score and a top-5 guesses chart.

## How it works

- **Features**: character-level n-grams (1–3 chars) via `TfidfVectorizer(analyzer="char_wb")`.
  Character patterns generalize better than words for language ID — they work
  even on short text and catch language-specific letter combos (e.g. "sch" in
  German, "ção" in Portuguese).
- **Model**: Multinomial Naive Bayes — fast to train, works well on
  high-dimensional sparse text features.
- **UI**: Streamlit text box → prediction with confidence bar + bar chart of
  top 5 candidate languages.

## Ideas to extend

- Add a language-family confusion matrix to see which languages get mixed up.
- Swap Naive Bayes for `LogisticRegression` or `LinearSVC` for a small accuracy bump.
- Add a "paste a URL" mode that fetches and detects the page's language.
- Deploy on Streamlit Community Cloud for a shareable public link.
