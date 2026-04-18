"""
Phase 2 — Improved Sentiment Analyzer
=======================================
Improvements over Phase 1:
  1. Text preprocessing (lowercase, remove punctuation, stopwords, stemming)
  2. Compare multiple models: Naive Bayes vs Logistic Regression vs SVM
  3. Better evaluation with cross-validation

Key concepts:
  - Stopwords: Common words like "the", "is", "and" that add no meaning
  - Stemming: Reducing words to their root form ("running" → "run")
  - Cross-validation: Train/test on different data splits to get reliable accuracy
"""

import pandas as pd
import numpy as np
import re
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Download NLTK data (only needed once)
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download("stopwords", quiet=True)

# ──────────────────────────────────────────────
# STEP 1: Text Preprocessing
# ──────────────────────────────────────────────
print("=" * 60)
print("PHASE 2: Improved Sentiment Analyzer")
print("=" * 60)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Keep negation words — they flip sentiment!
# "not good" vs "good" have opposite meanings
negation_words = {"not", "no", "nor", "neither", "never", "nobody",
                  "nothing", "nowhere", "hardly", "barely", "scarcely",
                  "don", "don't", "doesn't", "didn't", "won't",
                  "wouldn't", "couldn't", "shouldn't", "isn't",
                  "aren't", "wasn't", "weren't"}
stop_words -= negation_words


def preprocess_text(text):
    """
    Clean and normalize text for better ML performance.

    Steps:
      1. Lowercase — "GREAT" and "great" should be treated the same
      2. Remove punctuation — "amazing!" → "amazing"
      3. Remove stopwords — drop "the", "is", etc. (but keep negations)
      4. Stemming — "running" → "run", "better" → "better"
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters and spaces
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)


# ──────────────────────────────────────────────
# STEP 2: Load and preprocess data
# ──────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "reviews.csv")
df = pd.read_csv(data_path)

print(f"\nOriginal text:     \"{df['review'].iloc[0]}\"")
df["cleaned_review"] = df["review"].apply(preprocess_text)
print(f"After preprocessing: \"{df['cleaned_review'].iloc[0]}\"")

X = df["cleaned_review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ──────────────────────────────────────────────
# STEP 3: Vectorize with improved TF-IDF
# ──────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),      # Unigrams + bigrams
    min_df=2,                # Ignore words appearing in fewer than 2 documents
    sublinear_tf=True,       # Apply log normalization to term frequency
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ──────────────────────────────────────────────
# STEP 4: Compare multiple models
# ──────────────────────────────────────────────
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM (Linear)": LinearSVC(random_state=42, max_iter=2000),
}

print(f"\n{'─' * 60}")
print("MODEL COMPARISON")
print(f"{'─' * 60}")

best_model_name = None
best_accuracy = 0
best_model = None

for name, model in models.items():
    # Cross-validation: train and test on 5 different data splits
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring="accuracy")

    # Also train on full training set and test on held-out test set
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{name}:")
    print(f"  Cross-validation accuracy: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")
    print(f"  Test accuracy:             {test_accuracy:.2%}")

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model_name = name
        best_model = model

print(f"\n{'─' * 60}")
print(f"BEST MODEL: {best_model_name} ({best_accuracy:.2%} accuracy)")
print(f"{'─' * 60}")

# Detailed report for the best model
y_pred_best = best_model.predict(X_test_tfidf)
print(f"\nDetailed Classification Report ({best_model_name}):")
print(classification_report(y_test, y_pred_best, target_names=["Negative", "Positive"]))

# ──────────────────────────────────────────────
# STEP 5: Save the best model
# ──────────────────────────────────────────────
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(models_dir, exist_ok=True)

joblib.dump(best_model, os.path.join(models_dir, "best_model.pkl"))
joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer_v2.pkl"))
print(f"\nBest model saved to models/ folder.")

# ──────────────────────────────────────────────
# STEP 6: Live predictions with the best model
# ──────────────────────────────────────────────
print(f"\n{'─' * 40}")
print("LIVE PREDICTIONS (Improved Model)")
print(f"{'─' * 40}")

test_reviews = [
    "This is an amazing product, absolutely love it!",
    "Terrible experience, waste of money.",
    "It's okay, nothing special but works fine.",
    "Not bad at all, quite decent actually.",
    "I would not recommend this to anyone.",
]

for review in test_reviews:
    cleaned = preprocess_text(review)
    review_tfidf = vectorizer.transform([cleaned])
    prediction = best_model.predict(review_tfidf)[0]
    label = "POSITIVE" if prediction == 1 else "NEGATIVE"
    print(f"  [{label}] → \"{review}\"")

# ──────────────────────────────────────────────
# STEP 7: Export full results for R
# ──────────────────────────────────────────────
all_predictions = best_model.predict(vectorizer.transform(df["cleaned_review"]))
results_df = pd.DataFrame({
    "review": df["review"],
    "cleaned_review": df["cleaned_review"],
    "actual": df["sentiment"],
    "predicted": all_predictions,
    "correct": (df["sentiment"] == all_predictions).astype(int),
})
results_path = os.path.join(os.path.dirname(__file__), "..", "data", "predictions_v2.csv")
results_df.to_csv(results_path, index=False)
print(f"\nFull results exported to data/predictions_v2.csv")
