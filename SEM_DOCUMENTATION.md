# SOFTWARE ENGINEERING MANAGEMENT DOCUMENTATION

## Sentiment Analysis of Product Reviews Using Machine Learning

**Date:** April 2026 | **Project Manager:** Pulkit Raj | **Subject:** Software Engineering Management

---

## 1. EXECUTIVE SUMMARY

| Field | Detail |
|-------|--------|
| **Project Name** | Sentiment Analysis of Product Reviews Using ML |
| **Domain** | Natural Language Processing (NLP) |
| **Duration** | 6 weeks |
| **Team Size** | 4 members |
| **Budget** | Rs. 0 (open-source stack, free-tier deployment) |
| **Delivery** | Web app + ML model + R visualizations + report |
| **Repository** | https://github.com/Pulkit2903/sentiment-analyzer |

**Objective:** Design, develop, and deploy an end-to-end sentiment analysis system that classifies product reviews as positive or negative using Python (ML), R (visualization), and Flask (web deployment).

**Scope:** Binary classification of English reviews using TF-IDF + classical ML (Naive Bayes, Logistic Regression, SVM), R-based ggplot2 charts, Flask REST API with frontend, deployed on Render.

**Out of Scope:** Multi-language, deep learning (BERT/GPT), mobile app, real-time social media streaming.

---

## 2. REQUIREMENTS ANALYSIS

### 2.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | Accept raw English text as input | High |
| FR-02 | Classify text as positive or negative | High |
| FR-03 | Preprocess text (lowercase, punctuation, stopwords, stemming) | High |
| FR-04 | Convert text to TF-IDF numerical features | High |
| FR-05 | Train 3 ML classifiers and auto-select the best | High |
| FR-06 | Report accuracy, precision, recall, F1-score, confusion matrix | High |
| FR-07 | Display confidence score with predictions | Medium |
| FR-08 | Web-based UI for entering reviews and viewing results | High |
| FR-09 | REST API endpoint for programmatic access | High |
| FR-10 | Generate statistical visualizations using R (ggplot2) | High |
| FR-11 | Export predictions as CSV for Python-R data exchange | Medium |
| FR-12 | Save trained models to disk for reuse | Medium |

### 2.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-01 | API response time | < 500ms |
| NFR-02 | Concurrent users without crashing | 10 users |
| NFR-03 | Cross-browser frontend rendering | Chrome, Firefox, Safari |
| NFR-04 | Modular code with separation of concerns | Separate folders per module |
| NFR-05 | Python 3.9+ compatibility | Portable |
| NFR-06 | Single-command deployability | `pip install && python build.py` |

### 2.3 Hardware & Software Requirements

| Component | Requirement |
|-----------|-------------|
| Processor | Intel i3 / Apple M1 or equivalent |
| RAM | 4 GB minimum |
| Storage | 500 MB free |
| GPU | Not required (CPU-only ML) |
| Python | 3.9+ |
| R | 4.0+ (for visualization only) |
| OS | macOS / Windows 10+ / Ubuntu 20.04+ |

---

## 3. FEASIBILITY STUDY

| Type | Assessment | Verdict |
|------|-----------|---------|
| **Technical** | All algorithms available in scikit-learn; datasets publicly available; Python + R ecosystems mature | Feasible |
| **Economic** | Rs. 0 total — all open-source software, free-tier hosting, public datasets, existing hardware | Feasible |
| **Operational** | Single webpage UI, no user training needed, automated deployment via build.py | Feasible |
| **Schedule** | 6-week timeline with incremental deliverables per phase | Feasible |

---

## 4. WORK BREAKDOWN STRUCTURE (WBS)

```
1.0 SENTIMENT ANALYSIS PROJECT
├── 1.1 PROJECT PLANNING (8 hrs)
│   ├── 1.1.1 Requirements gathering
│   ├── 1.1.2 Tech stack selection
│   └── 1.1.3 Role assignment and scheduling
├── 1.2 DATASET PREPARATION (4 hrs)
│   ├── 1.2.1 Create 50 labeled reviews (25 pos, 25 neg)
│   └── 1.2.2 Write generation script, validate balance
├── 1.3 PHASE 1 — BASELINE MODEL (12 hrs)
│   ├── 1.3.1 Train/test split (80/20, stratified)
│   ├── 1.3.2 TF-IDF vectorization
│   ├── 1.3.3 Naive Bayes training and evaluation
│   └── 1.3.4 Save model with joblib
├── 1.4 PHASE 2 — IMPROVED MODEL (16 hrs)
│   ├── 1.4.1 Text preprocessing (lowercase, stopwords, stemming)
│   ├── 1.4.2 Train NB, Logistic Regression, SVM
│   ├── 1.4.3 5-fold cross-validation comparison
│   └── 1.4.4 Auto-select and save best model
├── 1.5 R VISUALIZATION (8 hrs)
│   ├── 1.5.1 Sentiment distribution, accuracy, confusion matrix charts
│   └── 1.5.2 Reticulate integration example
├── 1.6 WEB APPLICATION (12 hrs)
│   ├── 1.6.1 Flask API (POST /api/predict)
│   ├── 1.6.2 Frontend (HTML/CSS/JS, dark theme)
│   └── 1.6.3 End-to-end local testing
├── 1.7 DEPLOYMENT (6 hrs)
│   ├── 1.7.1 Git init, GitHub push
│   ├── 1.7.2 Render configuration and deploy
│   └── 1.7.3 Debug and verify live URL
├── 1.8 TESTING (8 hrs)
└── 1.9 DOCUMENTATION (10 hrs)

TOTAL EFFORT: 84 person-hours
```

---

## 5. SCHEDULING CHARTS

### 5.1 Gantt Chart

```
Task                       W1      W2      W3      W4      W5      W6
───────────────────────── ─────── ─────── ─────── ─────── ─────── ───────
1.1 Planning              ██████
1.2 Dataset               ░░░███
1.3 Phase 1 (Baseline)            ██████
1.4 Phase 2 (Improved)            ░░░███  ██████
1.5 R Visualization                               ██████
1.6 Web Application                               ░░░███  ██████
1.7 Deployment                                            ░░░███
1.8 Testing                                               ███░░░  ███
1.9 Documentation                                                 ██████
```

### 5.2 Milestones

| Milestone | Date | Deliverable |
|-----------|------|-------------|
| M1: Kickoff | March 09 | Project plan finalized |
| M2: Dataset Ready | March 15 | `reviews.csv` — 50 balanced reviews |
| M3: Baseline Model | March 22 | TF-IDF + Naive Bayes trained |
| M4: Improved Model | March 29 | 3 models compared, best saved |
| M5: Visualizations | April 05 | 4 ggplot2 charts generated |
| M6: Web App | April 12 | API + frontend working locally |
| M7: Deployed | April 15 | Live on Render |
| M8: Delivery | April 19 | Final submission |

### 5.3 Critical Path

**Path:** Planning → Dataset → Phase 1 → Phase 2 → Web App → Deployment → Testing → Delivery

**Critical Path Duration:** 66 person-hours. R Visualization and Documentation have 6 hrs float each (non-critical).

---

## 6. PROJECT ESTIMATION AND METRICS

### 6.1 Size Estimation (LOC)

| Module | LOC |
|--------|----:|
| Dataset Generation | 42 |
| Phase 1 Baseline | 78 |
| Phase 2 Improved | 102 |
| Flask API | 82 |
| R Scripts | 92 |
| Frontend (HTML + CSS) | 214 |
| Build Script | 28 |
| **TOTAL** | **638** |

### 6.2 COCOMO Estimation (Organic Mode)

```
KLOC  = 0.638
Effort = 2.4 × (0.638)^1.05 = 1.50 person-months
Time   = 2.5 × (1.50)^0.38  = 2.93 months
Staff  = 1.50 / 2.93         = 0.51 ≈ 1 person
```

Our 6-week timeline with 4 part-time members is well within this estimate.

### 6.3 Function Point Analysis

| Type | Count | Weight | FP |
|------|------:|-------:|---:|
| External Inputs | 2 | 3 | 6 |
| External Outputs | 3 | 4 | 12 |
| Internal Logical Files | 3 | 7 | 21 |
| External Interface Files | 1 | 5 | 5 |
| External Inquiries | 1 | 3 | 3 |
| **UFP** | | | **47** |

VAF = 0.65 + (0.01 × 30) = 0.95 → **Adjusted FP = 45**

### 6.4 Quality Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| API response time | < 500ms | ~50ms |
| Model accuracy (sample) | > 50% | 70% (SVM) |
| Defect density | < 5/KLOC | 3.13/KLOC |
| Productivity | — | 7.6 LOC/person-hr |

---

## 7. FINANCIAL PLANNING

### 7.1 Cost Breakdown

| Category | Cost |
|----------|------|
| Software (Python, R, scikit-learn, Flask, VS Code, Git) | Rs. 0 (open-source) |
| Hardware (existing laptops) | Rs. 0 |
| Cloud hosting (Render free tier) | Rs. 0 |
| Dataset (public / self-created) | Rs. 0 |
| Human resources (academic project) | Rs. 0 |
| **TOTAL** | **Rs. 0** |

### 7.2 Production Scale-Up Budget (If Required)

| Item | Monthly | Annual |
|------|---------|--------|
| Render Pro (always-on) | Rs. 600 | Rs. 7,200 |
| Custom domain | Rs. 80 | Rs. 960 |
| AWS S3 storage | Rs. 100 | Rs. 1,200 |
| **Total** | **Rs. 780** | **Rs. 9,360** |

---

## 8. RESOURCE ALLOCATION

### 8.1 RACI Matrix

| Task | Pulkit (Lead) | Member 2 | Member 3 | Member 4 |
|------|:-:|:-:|:-:|:-:|
| Planning & coordination | **R** | C | C | C |
| Dataset preparation | **R** | A | I | I |
| Phase 1 (Naive Bayes) | A | **R** | I | I |
| Phase 2 (Preprocessing) | **R** | A | I | I |
| Phase 2 (Model comparison) | A | **R** | C | I |
| R Visualization | I | I | **R** | A |
| Flask API | **R** | I | A | C |
| Frontend (HTML/CSS/JS) | I | I | C | **R** |
| Deployment | **R** | A | I | I |
| Testing | C | C | **R** | A |
| Documentation | **R** | A | C | C |

**R** = Responsible, **A** = Accountable, **C** = Consulted, **I** = Informed

### 8.2 Effort Distribution

| Member | Role | Hours | % |
|--------|------|:-----:|:---:|
| Pulkit Raj | Project Lead, ML Engineer, DevOps | 30 | 35.7% |
| Member 2 | ML Engineer, Backend | 22 | 26.2% |
| Member 3 | R Programmer, Tester | 16 | 19.0% |
| Member 4 | Frontend Developer | 16 | 19.0% |
| **Total** | | **84** | **100%** |

---

## 9. TEAM ORGANISATION

```
                ┌──────────────────┐
                │  Faculty Guide   │
                │  (Supervisor)    │
                └────────┬─────────┘
                         │
                ┌────────▼─────────┐
                │   Pulkit Raj     │
                │   Project Lead   │
                └────────┬─────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
   ┌─────▼─────┐  ┌─────▼─────┐  ┌──────▼─────┐
   │ Member 2  │  │ Member 3  │  │ Member 4   │
   │ ML Engg   │  │ R + Test  │  │ Frontend   │
   └───────────┘  └───────────┘  └────────────┘
```

| Channel | Purpose | Frequency |
|---------|---------|-----------|
| WhatsApp Group | Daily updates, quick questions | Daily |
| In-person Meeting | Sprint review, code review | Weekly (1 hr) |
| GitHub Issues/PRs | Bug tracking, code review | As needed |
| Email to Faculty | Progress reports | Bi-weekly |

---

## 10. RISK MANAGEMENT

| Risk ID | Description | Prob. | Impact | Mitigation |
|---------|-------------|:-----:|:------:|------------|
| R-01 | Small dataset → low accuracy | High | Medium | Document benchmark comparisons with IMDB dataset |
| R-02 | Dependency version conflicts on deploy | Medium | High | Use version ranges, not exact pins |
| R-03 | Team member unavailable | Medium | Medium | Cross-train on adjacent modules |
| R-04 | Render cold starts (30s delay) | High | Low | Document in README; acceptable for academic use |
| R-05 | NLTK download fails (SSL issues) | Medium | High | Add SSL bypass in build script |
| R-06 | Scope creep (deep learning, multi-language) | Medium | Medium | Strict scope; move extras to "Future Scope" |

**Risks Encountered:**
- **R-02 occurred:** `scikit-learn==1.4.2` pulled incompatible `numpy` on Render → Fixed by switching to version ranges.
- **R-05 occurred:** NLTK stopwords download failed due to macOS SSL → Fixed with `ssl._create_unverified_context`.

---

## 11. QUALITY ASSURANCE

### Test Cases

| TC | Scenario | Expected Result | Status |
|----|----------|-----------------|--------|
| TC-01 | Positive review input | `{"sentiment": "positive"}` | Pass |
| TC-02 | Negative review input | `{"sentiment": "negative"}` | Pass |
| TC-03 | Empty text input | HTTP 400, error message | Pass |
| TC-04 | Missing text field | HTTP 400, error message | Pass |
| TC-05 | Negation: "Not good" | `{"sentiment": "negative"}` | Pass |
| TC-06 | Frontend renders in browser | Page loads, buttons work | Pass |
| TC-07 | Model files created after training | `.pkl` files in `models/` | Pass |
| TC-08 | R script executes | 4 PNG charts generated | Pass |
| TC-09 | Deployment build succeeds | "Build successful" in Render logs | Pass |

---

## 12. CONFIGURATION MANAGEMENT

| Item | Detail |
|------|--------|
| VCS | Git + GitHub |
| Repository | https://github.com/Pulkit2903/sentiment-analyzer |
| Branch Strategy | Single `main` branch |

**Release History:**

| Version | Commit | Description |
|---------|--------|-------------|
| v1.0.0 | `7942c96` | Initial release — full pipeline + API + frontend |
| v1.0.1 | `fc93986` | Fix: version ranges in requirements.txt |
| v1.0.2 | `4562c08` | Add project report (REPORT.md) |

---

## 13. PROJECT CLOSURE

### Deliverables

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | Source code (GitHub) | Delivered |
| 2 | Trained ML models (.pkl) | Delivered |
| 3 | R visualization scripts | Delivered |
| 4 | Flask API + web frontend | Delivered |
| 5 | Live deployed application | Delivered |
| 6 | README.md, REPORT.md, SEM docs | Delivered |

### Lessons Learned

1. Use version ranges in `requirements.txt` — exact pins break across environments.
2. Handle SSL certificates explicitly in CI/CD build scripts.
3. Port 5000 conflicts with macOS AirPlay — use alternatives (5001).
4. Small datasets demonstrate pipelines but not model performance — include benchmarks.
5. Retain negation words during stopword removal for sentiment accuracy.

### Project Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Project Lead | Pulkit Raj | _____________ | __ / __ / 2026 |
| Team Member 2 | _____________ | _____________ | __ / __ / 2026 |
| Team Member 3 | _____________ | _____________ | __ / __ / 2026 |
| Team Member 4 | _____________ | _____________ | __ / __ / 2026 |
| Faculty Guide | _____________ | _____________ | __ / __ / 2026 |

---

*Project Repository: https://github.com/Pulkit2903/sentiment-analyzer*
