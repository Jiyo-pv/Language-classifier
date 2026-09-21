# Language Detection System

A machine learning web application that detects the language of text or text extracted from images using OCR.

## Live Demo

https://linear-svm-language-classifier.streamlit.app/

## Features

- Detects 17 languages
- Text language detection
- Image language detection using Tesseract OCR
- Confidence score and top 5 predictions
- Streamlit web interface

## Technologies

- Python
- Scikit-learn
- Streamlit
- Pandas
- Pillow
- Tesseract OCR
- Joblib

## Project Structure

```text
language-detection/
├── app.py
├── train.py
├── model.joblib
├── requirements.txt
├── packages.txt
├── data/
│   └── Language Detection.csv
└── README.md
```

## Dataset

Kaggle Language Detection Dataset

https://www.kaggle.com/datasets/basilb2s/language-detection

Place the dataset in:

```text
data/Language Detection.csv
```

## Installation

```bash
git clone https://github.com/Jiyo-pv/Language-classifier
cd language-detection
pip install -r requirements.txt
```

## Train the Model

```bash
python train.py
```

## Run the Application

```bash
streamlit run app.py
```

Open in your browser:

```text
http://localhost:8501
```

## Model Performance

| Metric | Value |
|---------|------:|
| Accuracy | 99.03% |
| Precision | 0.99 |
| Recall | 0.99 |
| F1 Score | 0.99 |
