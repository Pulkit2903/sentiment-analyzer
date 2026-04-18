"""
Phase 4 — Flask API for Sentiment Analysis
============================================
Exposes the trained model as a REST API.

Endpoints:
  GET  /                → Serves the frontend UI
  POST /api/predict     → Accepts JSON {"text": "..."}, returns sentiment

Run locally:
  python api/app.py
  Open http://localhost:5000 in your browser
"""

import os
import sys
import re
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download("stopwords", quiet=True)

# ── Setup ──
# Resolve project root (works both locally and in production)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static"),
)
CORS(app)

# ── Load the trained model ──
models_dir = os.path.join(PROJECT_ROOT, "models")

model_path = os.path.join(models_dir, "best_model.pkl")
vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer_v2.pkl")

# Fall back to Phase 1 model if Phase 2 hasn't been run yet
if not os.path.exists(model_path):
    model_path = os.path.join(models_dir, "naive_bayes_model.pkl")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")

if not os.path.exists(model_path):
    print("ERROR: No trained model found. Run phase1_basic.py or phase2_improved.py first.")
    sys.exit(1)

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
print(f"Loaded model from {model_path}")

# ── Preprocessing (same as Phase 2) ──
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
negation_words = {"not", "no", "nor", "neither", "never", "nobody",
                  "nothing", "nowhere", "hardly", "barely", "scarcely",
                  "don", "don't", "doesn't", "didn't", "won't",
                  "wouldn't", "couldn't", "shouldn't", "isn't",
                  "aren't", "wasn't", "weren't"}
stop_words -= negation_words

# Check if the loaded vectorizer was trained with preprocessing
# (Phase 2 vectorizer expects preprocessed text, Phase 1 does not)
uses_preprocessing = "tfidf_vectorizer_v2" in vectorizer_path


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)


# ── Routes ──
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400

    # Preprocess if using Phase 2 model
    processed_text = preprocess_text(text) if uses_preprocessing else text

    # Predict
    text_tfidf = vectorizer.transform([processed_text])
    prediction = model.predict(text_tfidf)[0]

    # Get confidence if the model supports predict_proba
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(text_tfidf)[0]
        confidence = round(float(max(proba)) * 100, 1)

    result = {
        "text": text,
        "sentiment": "positive" if prediction == 1 else "negative",
        "label": int(prediction),
    }
    if confidence is not None:
        result["confidence"] = confidence

    return jsonify(result)


if __name__ == "__main__":
    print("\n  Sentiment Analysis API running at http://localhost:5000\n")
    app.run(debug=True, port=5000)
