"""
generate_dataset.py — Create a sample sentiment dataset for training.

In a real project you'd use a dataset like:
  - IMDB Movie Reviews (50K reviews): https://ai.stanford.edu/~amaas/data/sentiment/
  - Kaggle Sentiment140 (1.6M tweets): https://www.kaggle.com/datasets/kazanova/sentiment140

For this project, we create a small labeled dataset so you can run everything
immediately without downloading anything.
"""

import pandas as pd
import os

# Each tuple is (review_text, sentiment_label)
# 1 = positive, 0 = negative
reviews = [
    # ── Positive reviews ──
    ("This product is absolutely amazing, I love it!", 1),
    ("Great quality and fast shipping. Very satisfied.", 1),
    ("Best purchase I've made this year. Highly recommend!", 1),
    ("The customer service was excellent and very helpful.", 1),
    ("Wonderful experience, will definitely buy again.", 1),
    ("Perfect fit and great material. Exceeded expectations.", 1),
    ("I'm really impressed with the quality of this item.", 1),
    ("Outstanding performance and great value for money.", 1),
    ("Loved the packaging and the product works perfectly.", 1),
    ("Five stars! This is exactly what I was looking for.", 1),
    ("Incredible product, works like a charm every time.", 1),
    ("Super happy with my purchase, thank you so much!", 1),
    ("The quality is top-notch, better than expected.", 1),
    ("Fantastic item, my whole family enjoys using it.", 1),
    ("Very well made and durable. Worth every penny.", 1),
    ("Excellent build quality and beautiful design.", 1),
    ("This exceeded all my expectations. Truly remarkable.", 1),
    ("Amazing value, performs better than products twice the price.", 1),
    ("So glad I bought this, it makes my life easier.", 1),
    ("A must-have product. I've already recommended it to friends.", 1),
    ("Smooth transaction and the product is just wonderful.", 1),
    ("I've tried many similar products but this one is the best.", 1),
    ("Pleasantly surprised by the quality at this price point.", 1),
    ("Works exactly as described. No complaints at all.", 1),
    ("Quick delivery and the item is in perfect condition.", 1),

    # ── Negative reviews ──
    ("Terrible quality, broke after one day of use.", 0),
    ("Worst purchase ever. Complete waste of money.", 0),
    ("Very disappointed with this product. Not as described.", 0),
    ("The item arrived damaged and customer support was unhelpful.", 0),
    ("Poor quality material, feels very cheap and flimsy.", 0),
    ("Does not work as advertised. I want a refund.", 0),
    ("Extremely slow shipping and the product is defective.", 0),
    ("Save your money, this product is absolute garbage.", 0),
    ("Horrible experience from start to finish. Never again.", 0),
    ("The product stopped working after just one week.", 0),
    ("Cheaply made junk. Falls apart immediately.", 0),
    ("Completely useless product, don't waste your time.", 0),
    ("Misleading description. The actual product is nothing like the photos.", 0),
    ("Arrived late and broken. Worst online shopping experience.", 0),
    ("Terrible customer service, they refused to help me.", 0),
    ("The quality is awful, I regret buying this.", 0),
    ("Not worth even half the price. Total disappointment.", 0),
    ("Product malfunctioned on the first use. Very frustrating.", 0),
    ("I would give zero stars if I could. Avoid this seller.", 0),
    ("Packaging was damaged and the item inside was scratched.", 0),
    ("Doesn't fit as described, sizing is completely off.", 0),
    ("Waited three weeks for delivery and got the wrong item.", 0),
    ("This is a scam. The product is nothing like advertised.", 0),
    ("Stopped working after two days. No durability at all.", 0),
    ("Regret this purchase entirely. Wish I could return it.", 0),
]

def generate():
    """Save the dataset as a CSV file in the data/ folder."""
    df = pd.DataFrame(reviews, columns=["review", "sentiment"])

    # Shuffle so positive and negative aren't grouped together
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "reviews.csv")
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to {output_path}")
    print(f"Total reviews: {len(df)}")
    print(f"Positive: {df['sentiment'].sum()}")
    print(f"Negative: {len(df) - df['sentiment'].sum()}")
    return df

if __name__ == "__main__":
    generate()
