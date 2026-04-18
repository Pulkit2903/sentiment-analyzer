"""
build.py — Runs during deployment to prepare the app.

This script:
  1. Downloads NLTK stopwords data
  2. Generates the dataset (if not present)
  3. Trains the models (Phase 1 + Phase 2)

Run automatically by Render during build step.
"""

import ssl
import os
import subprocess
import sys

# Fix SSL for NLTK downloads in CI/CD environments
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Ensure we're running from the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

print("=" * 50)
print("BUILD STEP 1: Download NLTK data")
print("=" * 50)
import nltk
nltk.download("stopwords", quiet=False)

print("\n" + "=" * 50)
print("BUILD STEP 2: Generate dataset")
print("=" * 50)
subprocess.run([sys.executable, "python/generate_dataset.py"], check=True)

print("\n" + "=" * 50)
print("BUILD STEP 3: Train Phase 1 model")
print("=" * 50)
subprocess.run([sys.executable, "python/phase1_basic.py"], check=True)

print("\n" + "=" * 50)
print("BUILD STEP 4: Train Phase 2 model")
print("=" * 50)
subprocess.run([sys.executable, "python/phase2_improved.py"], check=True)

print("\n" + "=" * 50)
print("BUILD COMPLETE")
print("=" * 50)
