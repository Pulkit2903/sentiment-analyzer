# ============================================================
# Alternative: Using reticulate to call Python from R
# ============================================================
# The `reticulate` package lets you run Python code directly
# inside R — no CSV export needed.
#
# Install: install.packages("reticulate")
#
# This is useful when you want a tighter integration,
# but CSV-based exchange (as in visualize_sentiment.R) is
# simpler and better for beginners.
# ============================================================

library(reticulate)

# Point to your Python installation
# use_python("/usr/local/bin/python3")  # Uncomment and adjust path

# Import Python modules
sklearn <- import("sklearn")
pd <- import("pandas")
joblib <- import("joblib")

# Load the trained model and vectorizer from Python
model <- joblib$load("models/best_model.pkl")
vectorizer <- joblib$load("models/tfidf_vectorizer_v2.pkl")

# Function to predict sentiment from R
predict_sentiment <- function(text) {
  text_tfidf <- vectorizer$transform(list(text))
  prediction <- model$predict(text_tfidf)
  return(ifelse(prediction == 1, "Positive", "Negative"))
}

# Try it out
reviews <- c(
  "This product is wonderful and amazing!",
  "Terrible quality, completely broken.",
  "Average product, does the job."
)

cat("Sentiment predictions from R (using Python model via reticulate):\n\n")
for (review in reviews) {
  sentiment <- predict_sentiment(review)
  cat(sprintf("  [%s] → \"%s\"\n", sentiment, review))
}
