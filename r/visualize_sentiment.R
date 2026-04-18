# ============================================================
# R Visualization for Sentiment Analysis Results
# ============================================================
# This script reads the prediction results from Python and
# creates professional visualizations using ggplot2.
#
# How Python ↔ R integration works:
#   - Python exports results as CSV (data/predictions_v2.csv)
#   - R reads the CSV and creates plots
#   - This is the simplest and most portable approach
#   - Alternative: use the `reticulate` package to call Python from R directly
#
# Install required packages (run once):
#   install.packages(c("ggplot2", "dplyr", "gridExtra"))
# ============================================================

library(ggplot2)
library(dplyr)

# ── Load the predictions from Python ──
data <- read.csv("data/predictions_v2.csv", stringsAsFactors = FALSE)

# Map 0/1 to readable labels
data$actual_label <- ifelse(data$actual == 1, "Positive", "Negative")
data$predicted_label <- ifelse(data$predicted == 1, "Positive", "Negative")
data$correct_label <- ifelse(data$correct == 1, "Correct", "Incorrect")

cat("Loaded", nrow(data), "reviews\n")
cat("Accuracy:", mean(data$correct) * 100, "%\n\n")

# ── Plot 1: Sentiment Distribution ──
p1 <- ggplot(data, aes(x = actual_label, fill = actual_label)) +
  geom_bar(width = 0.6, show.legend = FALSE) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5, size = 5) +
  scale_fill_manual(values = c("Negative" = "#e74c3c", "Positive" = "#2ecc71")) +
  labs(
    title = "Sentiment Distribution in Dataset",
    x = "Sentiment",
    y = "Number of Reviews"
  ) +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggsave("data/plot_sentiment_distribution.png", p1, width = 8, height = 6, dpi = 150)
cat("Saved: data/plot_sentiment_distribution.png\n")

# ── Plot 2: Model Accuracy (Correct vs Incorrect) ──
p2 <- ggplot(data, aes(x = correct_label, fill = correct_label)) +
  geom_bar(width = 0.6, show.legend = FALSE) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5, size = 5) +
  scale_fill_manual(values = c("Correct" = "#2ecc71", "Incorrect" = "#e74c3c")) +
  labs(
    title = "Model Prediction Accuracy",
    x = "Prediction Result",
    y = "Count"
  ) +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggsave("data/plot_accuracy.png", p2, width = 8, height = 6, dpi = 150)
cat("Saved: data/plot_accuracy.png\n")

# ── Plot 3: Confusion Matrix Heatmap ──
confusion <- data %>%
  group_by(actual_label, predicted_label) %>%
  summarise(count = n(), .groups = "drop")

p3 <- ggplot(confusion, aes(x = predicted_label, y = actual_label, fill = count)) +
  geom_tile(color = "white", linewidth = 2) +
  geom_text(aes(label = count), size = 8, fontface = "bold", color = "white") +
  scale_fill_gradient(low = "#3498db", high = "#2c3e50") +
  labs(
    title = "Confusion Matrix",
    x = "Predicted Sentiment",
    y = "Actual Sentiment",
    fill = "Count"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "none"
  )

ggsave("data/plot_confusion_matrix.png", p3, width = 8, height = 6, dpi = 150)
cat("Saved: data/plot_confusion_matrix.png\n")

# ── Plot 4: Review Length Distribution by Sentiment ──
data$review_length <- nchar(data$review)

p4 <- ggplot(data, aes(x = review_length, fill = actual_label)) +
  geom_histogram(bins = 15, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("Negative" = "#e74c3c", "Positive" = "#2ecc71")) +
  labs(
    title = "Review Length Distribution by Sentiment",
    x = "Review Length (characters)",
    y = "Count",
    fill = "Sentiment"
  ) +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggsave("data/plot_review_length.png", p4, width = 8, height = 6, dpi = 150)
cat("Saved: data/plot_review_length.png\n")

cat("\nAll plots saved successfully!\n")
