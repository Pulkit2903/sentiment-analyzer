# Sentiment Analyzer

An end-to-end sentiment analysis system that classifies product reviews as **Positive** or **Negative**. Built with Python for machine learning, R for statistical visualization, and Flask for web deployment.

---

## Table of Contents

1. [Tech Stack](#tech-stack)
2. [Why This Stack (and Why Not Others)](#why-this-stack-and-why-not-others)
3. [Project Structure](#project-structure)
4. [How It Works: Input-to-Output Pipeline](#how-it-works-input-to-output-pipeline)
5. [Setup and Run Instructions](#setup-and-run-instructions)
6. [Using a Real Dataset](#using-a-real-dataset)
7. [Deployment Options](#deployment-options)
8. [Future Scope (Resume-Level Extensions)](#future-scope-resume-level-extensions)
9. [Resume Bullet Point](#resume-bullet-point)

---

## Tech Stack

| Layer              | Technology           | Version  | Purpose                                      |
|--------------------|----------------------|----------|----------------------------------------------|
| **Language (ML)**  | Python               | 3.9+     | Core ML pipeline — preprocessing, training, prediction |
| **Language (Viz)** | R                    | 4.0+     | Statistical visualization and cross-language integration |
| **ML Library**     | scikit-learn         | 1.4+     | TF-IDF vectorization, classifiers (NB, LR, SVM), evaluation |
| **NLP Library**    | NLTK                 | 3.9+     | Stopword removal, Porter stemming            |
| **Data Handling**  | pandas, NumPy        | —        | DataFrame operations, numerical computation  |
| **Vectorization**  | TF-IDF (scikit-learn)| —        | Convert raw text into numerical feature vectors |
| **Visualization**  | ggplot2 (R)          | —        | Publication-quality charts (distribution, confusion matrix, etc.) |
| **R Integration**  | CSV exchange / reticulate | —   | Pass data between Python and R               |
| **API Framework**  | Flask                | 3.0+     | REST API to serve the trained model           |
| **Frontend**       | HTML + CSS + vanilla JS | —     | Browser-based UI for live predictions         |
| **Model Storage**  | joblib               | —        | Serialize trained models to `.pkl` files      |

---

## Why This Stack (and Why Not Others)

### Why Python for ML?

Python is the industry standard for machine learning. scikit-learn, TensorFlow, PyTorch — all major ML frameworks are Python-first. Using Python means:
- Largest ecosystem of ML libraries
- Maximum community support and tutorials
- Directly transferable to industry jobs

**Why not Java/C++?** — They lack the ML library ecosystem. You'd spend weeks writing what scikit-learn does in one line.

### Why scikit-learn (and not TensorFlow/PyTorch)?

| Factor             | scikit-learn              | TensorFlow / PyTorch          |
|--------------------|---------------------------|-------------------------------|
| Learning curve     | Beginner-friendly         | Steep, needs deep learning knowledge |
| Setup complexity   | `pip install`, done       | GPU drivers, CUDA, large downloads |
| Training time      | Seconds on CPU            | Minutes to hours, needs GPU   |
| Data requirement   | Works with small datasets | Needs thousands+ of samples   |
| Interpretability   | High — you can inspect weights | Black-box neural networks     |
| Use case           | Classical ML, tabular/text| Images, language models, complex patterns |

**Bottom line:** For a text classification task with structured features (TF-IDF), classical ML models (Naive Bayes, Logistic Regression, SVM) are faster, simpler, and equally accurate. Deep learning is overkill here — it shines when you need to learn complex representations (images, raw language), not when features are already well-engineered.

### Why TF-IDF (and not Bag of Words or Word2Vec)?

| Method        | How It Works                               | Weakness                                    |
|---------------|--------------------------------------------|---------------------------------------------|
| **Bag of Words** | Counts how many times each word appears  | Treats "the" (appears everywhere) the same as "amazing" (appears rarely) |
| **TF-IDF**    | Weighs words by importance — frequent in this doc but rare across all docs get higher scores | Doesn't capture word order or meaning |
| **Word2Vec**  | Maps words to dense vectors capturing semantic meaning | Needs large data, harder to implement, overkill for binary classification |

**TF-IDF is the sweet spot** — it's smarter than Bag of Words (downweighs common words) and simpler than Word2Vec (no neural networks needed). Perfect for a first project.

### Why Naive Bayes, Logistic Regression, and SVM?

These three classifiers represent three fundamentally different approaches:

| Model                | How It Decides                          | Strength                     | Weakness                     |
|----------------------|-----------------------------------------|------------------------------|------------------------------|
| **Naive Bayes**      | Calculates probability using Bayes' theorem, assumes word independence | Extremely fast, great baseline for text | Independence assumption is unrealistic |
| **Logistic Regression** | Finds a linear boundary between classes by learning feature weights | Interpretable, reliable, industry workhorse | Struggles with non-linear patterns |
| **SVM (Linear)**     | Finds the maximum-margin hyperplane separating the classes | Best accuracy on high-dimensional text data | Slower on very large datasets |

We train all three and automatically pick the best one. This is standard practice — no single model wins every time.

### Why R for Visualization (and not matplotlib/seaborn)?

| Factor           | R + ggplot2                     | Python + matplotlib             |
|------------------|---------------------------------|----------------------------------|
| Plot quality     | Publication-ready by default    | Needs heavy customization        |
| Grammar          | Declarative ("map x to color")  | Imperative ("draw line here")    |
| Statistical plots| Built for statistics            | General purpose                  |
| College requirement | Demonstrates cross-language skills | Same language as ML code     |

**The real reason:** This project specifically requires R integration to demonstrate that you can work across languages — a skill employers value. We use CSV as the data exchange format (simple, universal) and also show `reticulate` (R package that calls Python directly).

### Why Flask (and not FastAPI or Django)?

| Framework  | Best For                  | Why Not For This Project         |
|------------|---------------------------|----------------------------------|
| **Flask**  | Simple APIs, small apps   | Perfect fit — minimal boilerplate |
| **FastAPI**| Async APIs, auto docs     | More complex, async not needed here |
| **Django** | Full web apps with ORM    | Massive overkill for one endpoint |

Flask lets us serve the model in ~30 lines of code. That's the right tool for the job.

### Why Vanilla JS Frontend (and not React/Vue)?

This is a single-page UI with one text input and one button. React would add a build step, node_modules, and 200+ files for something that's 50 lines of JavaScript. The frontend is not the focus — the ML pipeline is.

---

## Project Structure

```
sentiment-analyzer/
│
├── data/                          # All data files
│   ├── reviews.csv                # Input dataset (50 labeled reviews)
│   ├── predictions.csv            # Phase 1 test set predictions
│   ├── predictions_v2.csv         # Phase 2 full predictions (used by R)
│   └── plot_*.png                 # Generated R visualizations
│
├── models/                        # Serialized trained models
│   ├── naive_bayes_model.pkl      # Phase 1 model
│   ├── tfidf_vectorizer.pkl       # Phase 1 vectorizer
│   ├── best_model.pkl             # Phase 2 best model (auto-selected)
│   └── tfidf_vectorizer_v2.pkl    # Phase 2 vectorizer (with preprocessing)
│
├── python/                        # Python ML scripts
│   ├── generate_dataset.py        # Creates the sample dataset
│   ├── phase1_basic.py            # Baseline: TF-IDF + Naive Bayes
│   └── phase2_improved.py         # Improved: preprocessing + model comparison
│
├── r/                             # R visualization scripts
│   ├── visualize_sentiment.R      # ggplot2 charts (4 plots)
│   └── reticulate_example.R       # Bonus: calling Python model from R
│
├── api/
│   └── app.py                     # Flask REST API server
│
├── templates/
│   └── index.html                 # Web frontend (Jinja2 template)
│
├── static/
│   └── style.css                  # Frontend styling (dark theme)
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## How It Works: Input-to-Output Pipeline

This section walks through the **complete journey** of a review — from raw text to a sentiment prediction displayed on screen.

### The Big Picture

```
User types a review
       |
       v
  ┌─────────────┐
  │  Frontend    │  (HTML/JS — templates/index.html)
  │  Browser UI  │
  └──────┬───────┘
         │  HTTP POST /api/predict  { "text": "Great product!" }
         v
  ┌─────────────┐
  │  Flask API   │  (api/app.py)
  └──────┬───────┘
         │
         v
  ┌─────────────────────────────────────────────────┐
  │  PREPROCESSING (NLP)                            │
  │                                                 │
  │  "Great product, I love it!"                    │
  │       │                                         │
  │       ├─ 1. Lowercase    → "great product..."   │
  │       ├─ 2. Remove punct → "great product i..." │
  │       ├─ 3. Remove stops → "great product love" │
  │       └─ 4. Stemming     → "great product love" │
  └──────────────────────┬──────────────────────────┘
                         │
                         v
  ┌─────────────────────────────────────────────────┐
  │  TF-IDF VECTORIZATION                           │
  │                                                 │
  │  "great product love"                           │
  │       │                                         │
  │       └─→ [0.0, 0.0, 0.52, ..., 0.71, 0.0]    │
  │           (sparse numerical vector)             │
  │           Each position = a word from vocab     │
  │           Each value = TF-IDF importance score  │
  └──────────────────────┬──────────────────────────┘
                         │
                         v
  ┌─────────────────────────────────────────────────┐
  │  ML MODEL PREDICTION                            │
  │                                                 │
  │  Trained classifier (NB / LR / SVM) receives    │
  │  the numerical vector and outputs:              │
  │       │                                         │
  │       ├─ Prediction: 1 (positive) or 0 (negative│)
  │       └─ Confidence: 87.3% (if model supports)  │
  └──────────────────────┬──────────────────────────┘
                         │
                         v
  ┌─────────────┐
  │  JSON       │  { "sentiment": "positive", "confidence": 87.3 }
  │  Response   │
  └──────┬──────┘
         │
         v
  ┌─────────────┐
  │  Frontend   │  Displays: [POSITIVE] with green styling
  │  Updates UI │
  └─────────────┘
```

### Step-by-Step Breakdown

#### Step 1: Data Collection (`python/generate_dataset.py`)

**What happens:** Creates a CSV file with 50 labeled product reviews (25 positive, 25 negative).

**Input:** None (hardcoded sample data)
**Output:** `data/reviews.csv`

```
review,sentiment
"This product is absolutely amazing, I love it!",1
"Terrible quality, broke after one day of use.",0
...
```

**Why this step matters:** Every ML model needs labeled training data. The label (1 or 0) is what the model learns to predict. In production, you'd use a real dataset like IMDB (50K reviews) or Sentiment140 (1.6M tweets).

---

#### Step 2: Train/Test Split (`phase1_basic.py` — Step 2)

**What happens:** Splits the 50 reviews into two groups:
- **Training set (80% = 40 reviews)** — The model learns from these
- **Test set (20% = 10 reviews)** — Used to evaluate accuracy on unseen data

**Why not use all data for training?** If the model sees all data during training, it might just memorize the answers instead of learning patterns. Testing on unseen data tells you how well it will perform on new reviews it has never seen before. This is called **generalization**.

**Key parameter — `stratify=y`:** Ensures both training and test sets have the same ratio of positive/negative reviews (50/50). Without this, random splitting might give you 90% positive in training and 90% negative in test, making results unreliable.

---

#### Step 3: Text Preprocessing (`phase2_improved.py` — Step 1)

**What happens:** Cleans raw text to reduce noise and help the model focus on meaningful words.

```
Input:  "This product is ABSOLUTELY amazing!! I love it."
                    │
   1. Lowercase     │  → "this product is absolutely amazing!! i love it."
   2. Remove punct  │  → "this product is absolutely amazing i love it"
   3. Remove stops  │  → "product absolutely amazing love"
   4. Stemming      │  → "product absolut amaz love"
                    │
Output: "product absolut amaz love"
```

**Why each step matters:**

| Step | What It Does | Why It Helps |
|------|-------------|-------------|
| Lowercase | "GREAT" → "great" | Without this, the model treats "Great" and "great" as two different words |
| Remove punctuation | "amazing!!" → "amazing" | Punctuation adds no predictive value for bag-of-words models |
| Remove stopwords | Drops "the", "is", "and", etc. | These words appear in every review regardless of sentiment — they're noise |
| Stemming | "running" → "run", "amazingly" → "amaz" | Reduces vocabulary size so the model can generalize better |

**Important design decision — keeping negation words:** We intentionally keep words like "not", "never", "don't" in the text even though they're technically stopwords. Why? Because "not good" has the opposite meaning of "good". Removing "not" would flip the sentiment entirely.

---

#### Step 4: TF-IDF Vectorization (`phase1_basic.py` — Step 3)

**What happens:** Converts cleaned text strings into numerical vectors that the ML model can process.

**The problem:** ML models understand numbers, not words. We need a way to represent text as a matrix of numbers.

**How TF-IDF works:**

```
TF (Term Frequency)  = How often a word appears in THIS review
                       "amazing amazing good" → amazing has TF = 2/3

IDF (Inverse Document Frequency) = How rare the word is ACROSS ALL reviews
                       "the" appears in every review → low IDF (not useful)
                       "amazing" appears in few reviews → high IDF (very useful)

TF-IDF = TF × IDF    → High score means: "this word is frequent in this
                        document but rare overall — it's probably important"
```

**The resulting matrix:**

```
              word_1  word_2  "amazing"  "terrible"  word_5  ...
Review 1:     [ 0.0,   0.0,    0.72,      0.0,       0.3,  ... ]
Review 2:     [ 0.1,   0.0,    0.0,       0.68,      0.0,  ... ]
Review 3:     [ 0.0,   0.5,    0.45,      0.0,       0.0,  ... ]
```

Each row is a review. Each column is a word. Each value is how important that word is to that review.

**Key parameters we use:**
- `max_features=5000` — Keep only the 5000 most informative words (reduces noise)
- `ngram_range=(1, 2)` — Capture both single words ("good") and two-word phrases ("not good")
- `min_df=2` — Ignore words that appear in fewer than 2 reviews (too rare to be useful)
- `sublinear_tf=True` — Use `1 + log(TF)` instead of raw TF to prevent long reviews from dominating

---

#### Step 5: Model Training (`phase2_improved.py` — Step 4)

**What happens:** Three different classifiers are trained on the TF-IDF matrix and compared:

**Naive Bayes (MultinomialNB):**
- Uses Bayes' theorem: P(positive | words) = P(words | positive) × P(positive) / P(words)
- Assumes each word contributes to sentiment independently ("naive" assumption)
- Extremely fast to train — good baseline

**Logistic Regression:**
- Learns a weight for each word: `score = w1×"amazing" + w2×"terrible" + w3×"good" + ...`
- If score > threshold → positive, else → negative
- Interpretable — you can inspect which words push toward positive/negative

**SVM (Support Vector Machine):**
- Finds the hyperplane that maximally separates positive and negative reviews in TF-IDF space
- Effective for high-dimensional data (5000 word features)
- Usually the most accurate for text classification

**Evaluation with 5-fold cross-validation:**
```
Split data into 5 parts:  [A] [B] [C] [D] [E]

Round 1: Train on [B,C,D,E], test on [A] → accuracy_1
Round 2: Train on [A,C,D,E], test on [B] → accuracy_2
Round 3: Train on [A,B,D,E], test on [C] → accuracy_3
Round 4: Train on [A,B,C,E], test on [D] → accuracy_4
Round 5: Train on [A,B,C,D], test on [E] → accuracy_5

Final = average(accuracy_1 ... accuracy_5)
```

This gives a more reliable accuracy estimate than a single train/test split.

---

#### Step 6: Model Serialization (Saving)

**What happens:** The best model and its TF-IDF vectorizer are saved to disk as `.pkl` files using `joblib`.

**Why save two files?**
1. **Model file** (`best_model.pkl`) — Contains the learned weights/probabilities
2. **Vectorizer file** (`tfidf_vectorizer_v2.pkl`) — Contains the learned vocabulary (which words map to which columns)

Both are needed at prediction time. The vectorizer must be the exact same one used during training — otherwise word-to-column mappings won't match and predictions will be garbage.

---

#### Step 7: Export Results to R (`phase2_improved.py` — Step 7)

**What happens:** Predictions for all 50 reviews are saved to `data/predictions_v2.csv` with columns:
- `review` — original text
- `cleaned_review` — preprocessed text
- `actual` — true label (ground truth)
- `predicted` — model's prediction
- `correct` — 1 if model was right, 0 if wrong

**Why CSV for Python→R data exchange?**
- Universal format — every language can read/write CSV
- Human-readable — you can open it in Excel to inspect
- No dependency — no special library needed
- Alternative: the `reticulate` R package can call Python code directly (shown in `r/reticulate_example.R`), but CSV is simpler for beginners

---

#### Step 8: R Visualization (`r/visualize_sentiment.R`)

**What happens:** R reads `predictions_v2.csv` and generates 4 publication-quality charts using ggplot2:

| Plot | What It Shows | File |
|------|--------------|------|
| Sentiment Distribution | Bar chart of positive vs negative review counts | `plot_sentiment_distribution.png` |
| Model Accuracy | How many predictions were correct vs incorrect | `plot_accuracy.png` |
| Confusion Matrix | Heatmap — true positives, false positives, etc. | `plot_confusion_matrix.png` |
| Review Length by Sentiment | Do positive/negative reviews differ in length? | `plot_review_length.png` |

---

#### Step 9: Flask API (`api/app.py`)

**What happens:** Loads the saved model and vectorizer, then exposes a REST endpoint.

**API contract:**

```
POST /api/predict
Content-Type: application/json

Request:  { "text": "This product is amazing!" }
Response: { "text": "This product is amazing!",
            "sentiment": "positive",
            "label": 1,
            "confidence": 87.3 }
```

**Internal flow when the API receives a request:**
1. Validate input (non-empty text)
2. Preprocess text (lowercase → remove punctuation → remove stopwords → stem)
3. Vectorize using the loaded TF-IDF vectorizer
4. Predict using the loaded ML model
5. Return JSON response

---

#### Step 10: Frontend (`templates/index.html` + `static/style.css`)

**What happens:** A browser-based UI where users type reviews and see sentiment results.

**How it works:**
1. User types text in the textarea
2. Clicks "Analyze Sentiment" (or presses Ctrl+Enter)
3. JavaScript sends a `fetch()` POST request to `/api/predict`
4. API returns JSON → JS updates the DOM with the result
5. Green box for positive, red box for negative

---

## Setup and Run Instructions

### Prerequisites

- Python 3.9 or higher
- R 4.0+ with `ggplot2` and `dplyr` packages (for visualization only)
- pip (Python package manager)

### Step-by-step

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Generate the sample dataset
python python/generate_dataset.py

# 3. Run Phase 1 — Basic model (TF-IDF + Naive Bayes)
python python/phase1_basic.py

# 4. Run Phase 2 — Improved model (preprocessing + NB vs LR vs SVM)
python python/phase2_improved.py

# 5. Generate R visualizations (requires R installed)
Rscript r/visualize_sentiment.R

# 6. Launch the web app
python api/app.py
# Open http://localhost:5000 in your browser
```

### Install R packages (one-time)

```r
install.packages(c("ggplot2", "dplyr"))
```

---

## Using a Real Dataset

The sample dataset has only 50 reviews, so accuracy will be low (30-70%). Replace it with a real dataset for production-level results (85-92% accuracy):

1. **IMDB Movie Reviews (50K)** — https://ai.stanford.edu/~amaas/data/sentiment/
2. **Kaggle Sentiment140 (1.6M tweets)** — https://www.kaggle.com/datasets/kazanova/sentiment140
3. **Amazon Product Reviews** — https://www.kaggle.com/datasets/bittlingmayer/amazonreviews

Save as `data/reviews.csv` with two columns: `review` (text) and `sentiment` (1 = positive, 0 = negative). Then re-run Phase 1 and Phase 2.

---

## Deployment Options

| Platform        | How To Deploy                                      | Cost    |
|-----------------|-----------------------------------------------------|---------|
| **Render**      | Connect GitHub repo, set start command `python api/app.py` | Free tier |
| **Heroku**      | Add `Procfile` with `web: python api/app.py`        | Free tier |
| **Docker**      | Containerize with `Dockerfile`, deploy anywhere     | Varies  |
| **AWS Lambda**  | Serverless — package model + API as Lambda function | Pay-per-use |

---

## Future Scope (Resume-Level Extensions)

### 1. Aspect-Based Sentiment Analysis

Instead of one sentiment per review, detect sentiment **per topic**:

```
Input:  "Great battery life but the camera is terrible"
Output: { "battery life": "positive", "camera": "negative" }
```

Use spaCy for noun-phrase extraction + per-aspect classification.

### 2. Real-Time Social Media Analysis

Stream live tweets/Reddit posts using the Twitter API or PRAW, classify sentiment in real-time, and display trends on a live dashboard (Plotly Dash or Streamlit).

### 3. Transformer-Based Model (BERT)

Fine-tune a DistilBERT model using Hugging Face `transformers` library. Compare its accuracy against the classical TF-IDF approach — this demonstrates understanding of both classical and deep learning methods.

---

## Resume Bullet Point

> **Sentiment Analysis Engine (Python, R, Flask)** — Engineered an end-to-end NLP pipeline for binary sentiment classification of product reviews using TF-IDF vectorization and classical ML models (Naive Bayes, Logistic Regression, SVM) with 5-fold cross-validation. Integrated R (ggplot2) for statistical visualization via CSV data exchange and deployed the model as a Flask REST API with a responsive web frontend.
