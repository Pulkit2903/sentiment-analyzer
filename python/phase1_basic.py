"""
Phase 1 — Basic Sentiment Analyzer
====================================
Pipeline: Raw Text → TF-IDF Vectorization → Naive Bayes Classifier

This is the simplest working version. We'll improve it in Phase 2.

Key concepts:
  1. TF-IDF (Term Frequency–Inverse Document Frequency)
     - Converts text into numerical vectors
     - Words that appear often in one document but rarely in others get higher scores
     - Better than simple word counts because it reduces the weight of common words

  2. Naive Bayes (MultinomialNB)
     - A probabilistic classifier based on Bayes' theorem
     - "Naive" because it assumes words are independent of each other
     - Fast, works well with text, and great as a baseline model
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ──────────────────────────────────────────────
# STEP 1: Load the dataset
# ──────────────────────────────────────────────
print("=" * 60)
print("PHASE 1: Basic Sentiment Analyzer")
print("=" * 60)

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "reviews.csv")
df = pd.read_csv(data_path)

print(f"\nDataset loaded: {len(df)} reviews")
print(f"Columns: {list(df.columns)}")
print(f"\nSample data:")
print(df.head())
print(f"\nSentiment distribution:")
print(df["sentiment"].value_counts().rename({1: "Positive", 0: "Negative"}))

# ──────────────────────────────────────────────
# STEP 2: Split into training and testing sets
# ──────────────────────────────────────────────
# We use 80% data for training, 20% for testing.
# stratify=y ensures both sets have the same ratio of positive/negative reviews.
X = df["review"]       # Features (the text)
y = df["sentiment"]    # Labels (0 or 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # For reproducibility
    stratify=y           # Keep class balance in both splits
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ──────────────────────────────────────────────
# STEP 3: Convert text to numbers using TF-IDF
# ──────────────────────────────────────────────
# The ML model can't read text — it needs numbers.
# TF-IDF creates a matrix where:
#   - Each row = one review
#   - Each column = one word from the vocabulary
#   - Each cell = the TF-IDF score of that word in that review
vectorizer = TfidfVectorizer(
    max_features=5000,   # Only keep the top 5000 most important words
    ngram_range=(1, 2),  # Use single words AND two-word phrases
)

X_train_tfidf = vectorizer.fit_transform(X_train)  # Learn vocabulary + transform
X_test_tfidf = vectorizer.transform(X_test)         # Transform only (no re-learning)

print(f"\nTF-IDF matrix shape: {X_train_tfidf.shape}")
print(f"(rows = reviews, columns = unique word features)")

# ──────────────────────────────────────────────
# STEP 4: Train the Naive Bayes model
# ──────────────────────────────────────────────
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
print("\nModel trained successfully!")

# ──────────────────────────────────────────────
# STEP 5: Evaluate the model
# ──────────────────────────────────────────────
y_pred = model.predict(X_test_tfidf)

print(f"\n{'─' * 40}")
print("MODEL EVALUATION")
print(f"{'─' * 40}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
print(f"Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ──────────────────────────────────────────────
# STEP 6: Save the model and vectorizer
# ──────────────────────────────────────────────
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(models_dir, exist_ok=True)

joblib.dump(model, os.path.join(models_dir, "naive_bayes_model.pkl"))
joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
print(f"\nModel and vectorizer saved to models/ folder.")

# ──────────────────────────────────────────────
# STEP 7: Try it on custom text
# ──────────────────────────────────────────────
print(f"\n{'─' * 40}")
print("LIVE PREDICTIONS")
print(f"{'─' * 40}")

test_reviews = [
    "This is an amazing product, absolutely love it!",
    "Terrible experience, waste of money.",
    "It's okay, nothing special but works fine.",
    "Brilliant quality and super fast delivery!",
    "Broke after two days. Very disappointed.",
]

for review in test_reviews:
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)[0]
    confidence = model.predict_proba(review_tfidf).max()
    label = "POSITIVE" if prediction == 1 else "NEGATIVE"
    print(f"  [{label}] (confidence: {confidence:.0%}) → \"{review}\"")

# ──────────────────────────────────────────────
# STEP 8: Export predictions for R visualization
# ──────────────────────────────────────────────
results_df = pd.DataFrame({
    "review": X_test.values,
    "actual": y_test.values,
    "predicted": y_pred,
})
results_path = os.path.join(os.path.dirname(__file__), "..", "data", "predictions.csv")
results_df.to_csv(results_path, index=False)
print(f"\nPredictions exported to data/predictions.csv (for R visualization)")
