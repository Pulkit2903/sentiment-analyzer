# SOFTWARE ENGINEERING MANAGEMENT DOCUMENTATION

## Sentiment Analysis of Product Reviews Using Machine Learning

**Document Version:** 1.0
**Date:** April 2026
**Project Manager:** Pulkit Raj
**Subject:** Software Engineering Management

---

## TABLE OF CONTENTS

1. Executive Summary
2. Requirements Analysis
3. Feasibility Study
4. Work Breakdown Structure (WBS)
5. Scheduling Charts
6. Project Estimation and Metrics
7. Financial Planning and Cost Estimation
8. Resource Allocation
9. Team Organisation
10. Risk Management
11. Quality Assurance Plan
12. Configuration Management
13. Communication Plan
14. Project Closure

---

## 1. EXECUTIVE SUMMARY

### 1.1 Project Overview

| Field | Detail |
|-------|--------|
| **Project Name** | Sentiment Analysis of Product Reviews Using Machine Learning |
| **Project Type** | Academic / R&D — AI/ML Application |
| **Domain** | Natural Language Processing (NLP) |
| **Duration** | 6 weeks |
| **Team Size** | 4 members |
| **Budget** | Rs. 0 (open-source stack, free-tier deployment) |
| **Delivery** | Web application + trained ML model + R visualizations + project report |
| **Deployment** | Render (free tier) — https://sentiment-analyzer-XXXX.onrender.com |
| **Repository** | https://github.com/Pulkit2903/sentiment-analyzer |

### 1.2 Project Objective

To design, develop, and deploy an end-to-end sentiment analysis system that classifies product reviews as positive or negative using classical machine learning techniques, with Python for the ML pipeline, R for statistical visualization, and Flask for web deployment.

### 1.3 Scope Statement

**In Scope:**
- Binary sentiment classification (positive/negative) of English product reviews
- Text preprocessing pipeline (lowercase, punctuation removal, stopword removal, stemming)
- TF-IDF vectorization and three ML classifiers (Naive Bayes, Logistic Regression, SVM)
- R-based statistical visualizations using ggplot2
- Flask REST API with responsive web frontend
- Deployment on Render (free tier)

**Out of Scope:**
- Multi-language sentiment analysis
- Multi-class sentiment (5-star rating prediction)
- Deep learning models (BERT, GPT)
- Mobile application
- Real-time streaming from social media APIs
- User authentication and session management

---

## 2. REQUIREMENTS ANALYSIS

### 2.1 Stakeholder Identification

| Stakeholder | Role | Interest |
|-------------|------|----------|
| Students (Development Team) | Developers, testers | Build the project, learn ML concepts |
| Faculty Guide | Supervisor, evaluator | Evaluate project quality, guide methodology |
| College Department | Academic institution | Ensure curriculum alignment |
| End Users | Product managers, analysts | Use the tool to classify reviews |

### 2.2 Functional Requirements

| ID | Requirement | Priority | Module |
|----|-------------|----------|--------|
| FR-01 | System shall accept raw English text as input | High | API |
| FR-02 | System shall classify text as positive or negative | High | ML Pipeline |
| FR-03 | System shall preprocess text (lowercase, remove punctuation, remove stopwords, stem) | High | Preprocessing |
| FR-04 | System shall convert text to numerical features using TF-IDF | High | Feature Extraction |
| FR-05 | System shall train at least 3 ML classifiers and select the best one | High | Model Training |
| FR-06 | System shall report accuracy, precision, recall, F1-score, and confusion matrix | High | Evaluation |
| FR-07 | System shall display confidence score with each prediction (where supported) | Medium | API |
| FR-08 | System shall provide a web-based UI for entering reviews and viewing results | High | Frontend |
| FR-09 | System shall expose a REST API endpoint for programmatic access | High | API |
| FR-10 | System shall generate statistical visualizations using R and ggplot2 | High | R Module |
| FR-11 | System shall export prediction results as CSV for cross-language data exchange | Medium | Data Export |
| FR-12 | System shall save trained models to disk for reuse without retraining | Medium | Serialization |

### 2.3 Non-Functional Requirements

| ID | Requirement | Category | Target |
|----|-------------|----------|--------|
| NFR-01 | API response time shall be under 500ms per request | Performance | < 500ms |
| NFR-02 | System shall handle concurrent requests without crashing | Reliability | 10 simultaneous users |
| NFR-03 | Frontend shall render correctly on desktop and mobile browsers | Usability | Chrome, Firefox, Safari |
| NFR-04 | Code shall be modular with clear separation of concerns | Maintainability | Separate folders per module |
| NFR-05 | All Python code shall be compatible with Python 3.9+ | Portability | Python 3.9, 3.10, 3.11, 3.12 |
| NFR-06 | System shall be deployable using a single build command | Deployability | `pip install && python build.py` |
| NFR-07 | Codebase shall include inline comments explaining key ML concepts | Documentation | Every function documented |
| NFR-08 | System shall not store or log user input text on the server | Privacy | No data persistence |

### 2.4 Hardware Requirements

| Component | Minimum Specification |
|-----------|----------------------|
| Processor | Intel i3 / Apple M1 or equivalent |
| RAM | 4 GB |
| Storage | 500 MB free disk space |
| Network | Internet connection (for package installation and deployment) |
| GPU | Not required (classical ML, CPU-only) |

### 2.5 Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.9+ | Core ML pipeline, API server |
| R | 4.0+ | Statistical visualization |
| pip | Latest | Python package management |
| Git | 2.30+ | Version control |
| VS Code / RStudio | Latest | Development environment |
| Chrome / Firefox | Latest | Testing the web frontend |
| Operating System | macOS / Windows 10+ / Ubuntu 20.04+ | Development and testing |

### 2.6 Requirements Traceability Matrix (RTM)

| Requirement ID | Design Module | Code File | Test Case | Status |
|----------------|---------------|-----------|-----------|--------|
| FR-01 | API Input Handler | `api/app.py:predict()` | TC-01 | Implemented |
| FR-02 | ML Classifier | `python/phase2_improved.py` | TC-02 | Implemented |
| FR-03 | Text Preprocessor | `python/phase2_improved.py:preprocess_text()` | TC-03 | Implemented |
| FR-04 | TF-IDF Vectorizer | `python/phase2_improved.py` | TC-04 | Implemented |
| FR-05 | Model Comparison | `python/phase2_improved.py` | TC-05 | Implemented |
| FR-06 | Evaluation Metrics | `python/phase1_basic.py`, `phase2_improved.py` | TC-06 | Implemented |
| FR-07 | Confidence Score | `api/app.py:predict()` | TC-07 | Implemented |
| FR-08 | Web Frontend | `templates/index.html`, `static/style.css` | TC-08 | Implemented |
| FR-09 | REST API | `api/app.py` | TC-09 | Implemented |
| FR-10 | R Visualization | `r/visualize_sentiment.R` | TC-10 | Implemented |
| FR-11 | CSV Export | `python/phase2_improved.py` | TC-11 | Implemented |
| FR-12 | Model Serialization | `python/phase2_improved.py` (joblib) | TC-12 | Implemented |

---

## 3. FEASIBILITY STUDY

### 3.1 Technical Feasibility

| Factor | Assessment | Verdict |
|--------|-----------|---------|
| Algorithm availability | TF-IDF, NB, LR, SVM well-established in scikit-learn | Feasible |
| Dataset availability | IMDB, Sentiment140, Amazon reviews freely available | Feasible |
| Language support | Python + R — both have mature ML/stats ecosystems | Feasible |
| Deployment platform | Render free tier supports Python web apps | Feasible |
| Team skill level | Basic Python + R knowledge sufficient; ML concepts learnable | Feasible |

**Conclusion:** Technically feasible. All tools are open-source, well-documented, and mature.

### 3.2 Economic Feasibility

| Item | Cost |
|------|------|
| Software licenses | Rs. 0 (all open-source) |
| Cloud hosting | Rs. 0 (Render free tier) |
| Dataset | Rs. 0 (public datasets) |
| Hardware | Rs. 0 (using existing laptops) |
| **Total** | **Rs. 0** |

**Conclusion:** Economically feasible. Zero-cost project using free tools and infrastructure.

### 3.3 Operational Feasibility

The system is designed for simplicity — a single webpage where users type text and get results. No training is required for end users. The deployment pipeline is automated through `build.py`, requiring no manual intervention after initial setup.

**Conclusion:** Operationally feasible.

### 3.4 Schedule Feasibility

The project is scoped for a 6-week timeline with well-defined phases. Each phase builds on the previous one and produces a working deliverable, allowing for incremental progress tracking.

**Conclusion:** Schedule feasible.

---

## 4. WORK BREAKDOWN STRUCTURE (WBS)

### 4.1 WBS Hierarchy

```
1.0 SENTIMENT ANALYSIS PROJECT
│
├── 1.1 PROJECT PLANNING
│   ├── 1.1.1 Requirements gathering and analysis
│   ├── 1.1.2 Feasibility study
│   ├── 1.1.3 Technology stack selection
│   ├── 1.1.4 Project schedule creation
│   └── 1.1.5 Role assignment
│
├── 1.2 DATASET PREPARATION
│   ├── 1.2.1 Research available datasets (IMDB, Sentiment140)
│   ├── 1.2.2 Create sample dataset (50 labeled reviews)
│   ├── 1.2.3 Write dataset generation script
│   ├── 1.2.4 Define CSV schema (review, sentiment)
│   └── 1.2.5 Validate dataset balance and quality
│
├── 1.3 ML PIPELINE — PHASE 1 (BASELINE)
│   ├── 1.3.1 Load and explore dataset
│   ├── 1.3.2 Train/test split (80/20, stratified)
│   ├── 1.3.3 Implement TF-IDF vectorization
│   ├── 1.3.4 Train Naive Bayes classifier
│   ├── 1.3.5 Evaluate (accuracy, classification report, confusion matrix)
│   ├── 1.3.6 Save model and vectorizer (joblib)
│   └── 1.3.7 Test with custom input reviews
│
├── 1.4 ML PIPELINE — PHASE 2 (IMPROVED)
│   ├── 1.4.1 Implement text preprocessing function
│   │   ├── 1.4.1.1 Lowercasing
│   │   ├── 1.4.1.2 Punctuation removal (regex)
│   │   ├── 1.4.1.3 Stopword removal with negation retention
│   │   └── 1.4.1.4 Porter stemming
│   ├── 1.4.2 Improve TF-IDF parameters (min_df, sublinear_tf, bigrams)
│   ├── 1.4.3 Train Logistic Regression classifier
│   ├── 1.4.4 Train SVM (LinearSVC) classifier
│   ├── 1.4.5 Implement 5-fold cross-validation
│   ├── 1.4.6 Compare all 3 models and auto-select best
│   ├── 1.4.7 Save best model
│   └── 1.4.8 Export results to CSV for R
│
├── 1.5 R VISUALIZATION — PHASE 3
│   ├── 1.5.1 Read prediction CSV in R
│   ├── 1.5.2 Sentiment distribution bar chart
│   ├── 1.5.3 Model accuracy chart
│   ├── 1.5.4 Confusion matrix heatmap
│   ├── 1.5.5 Review length distribution histogram
│   └── 1.5.6 Write reticulate integration example
│
├── 1.6 WEB APPLICATION — PHASE 4 & 5
│   ├── 1.6.1 Design Flask API architecture
│   ├── 1.6.2 Implement POST /api/predict endpoint
│   ├── 1.6.3 Implement model loading and preprocessing in API
│   ├── 1.6.4 Design frontend UI (HTML structure)
│   ├── 1.6.5 Style frontend (CSS — dark theme, responsive)
│   ├── 1.6.6 Implement JavaScript (fetch API, DOM updates)
│   ├── 1.6.7 Add example review buttons
│   └── 1.6.8 Test end-to-end locally
│
├── 1.7 DEPLOYMENT
│   ├── 1.7.1 Create .gitignore
│   ├── 1.7.2 Write requirements.txt with version ranges
│   ├── 1.7.3 Write build.py (automated training on deploy)
│   ├── 1.7.4 Create Procfile and render.yaml
│   ├── 1.7.5 Initialize Git repository
│   ├── 1.7.6 Push to GitHub
│   ├── 1.7.7 Configure Render service
│   ├── 1.7.8 Debug deployment errors
│   └── 1.7.9 Verify live deployment
│
├── 1.8 TESTING
│   ├── 1.8.1 Unit test — preprocessing function
│   ├── 1.8.2 Unit test — API endpoint (valid/invalid input)
│   ├── 1.8.3 Integration test — end-to-end prediction
│   ├── 1.8.4 UI test — browser rendering and interaction
│   └── 1.8.5 Performance test — response time under load
│
└── 1.9 DOCUMENTATION
    ├── 1.9.1 Write README.md (setup, usage, tech stack)
    ├── 1.9.2 Write REPORT.md (problem statement, methodology, results)
    ├── 1.9.3 Write SEM documentation (this document)
    ├── 1.9.4 Add inline code comments
    └── 1.9.5 Prepare presentation slides
```

### 4.2 WBS Dictionary

| WBS Code | Task Name | Description | Deliverable | Estimated Effort |
|----------|-----------|-------------|-------------|-----------------|
| 1.1 | Project Planning | Define scope, tech stack, roles, schedule | Project plan document | 8 person-hours |
| 1.2 | Dataset Preparation | Create and validate labeled review dataset | `data/reviews.csv` | 4 person-hours |
| 1.3 | Phase 1 Baseline | Basic TF-IDF + Naive Bayes pipeline | `phase1_basic.py`, model files | 12 person-hours |
| 1.4 | Phase 2 Improved | Preprocessing + model comparison | `phase2_improved.py`, best model | 16 person-hours |
| 1.5 | R Visualization | Statistical charts using ggplot2 | `.R` scripts, PNG plots | 8 person-hours |
| 1.6 | Web Application | Flask API + frontend | `app.py`, `index.html`, `style.css` | 12 person-hours |
| 1.7 | Deployment | GitHub + Render deployment | Live URL | 6 person-hours |
| 1.8 | Testing | Unit, integration, UI, performance tests | Test results | 8 person-hours |
| 1.9 | Documentation | README, report, SEM docs | `.md` files | 10 person-hours |
| | **TOTAL** | | | **84 person-hours** |

---

## 5. SCHEDULING CHARTS

### 5.1 Project Timeline (6 Weeks)

```
Week 1: March 09 – March 15, 2026
Week 2: March 16 – March 22, 2026
Week 3: March 23 – March 29, 2026
Week 4: March 30 – April 05, 2026
Week 5: April 06 – April 12, 2026
Week 6: April 13 – April 19, 2026
```

### 5.2 Gantt Chart

```
Task                          W1      W2      W3      W4      W5      W6
──────────────────────────── ─────── ─────── ─────── ─────── ─────── ───────
1.1 Project Planning         ██████
1.2 Dataset Preparation      ░░░███
1.3 Phase 1 (Baseline)               ██████
1.4 Phase 2 (Improved)               ░░░███  ██████
1.5 R Visualization                                  ██████
1.6 Web Application                                  ░░░███  ██████
1.7 Deployment                                               ░░░███
1.8 Testing                                                  ███░░░  ███
1.9 Documentation                                                    ██████

██ = Primary work period
░░ = Overlap / preparation
```

### 5.3 Milestone Chart

| Milestone | Date | Deliverable | Criteria |
|-----------|------|-------------|----------|
| M1: Project Kickoff | March 09 | Approved project plan | Scope, tech stack, roles finalized |
| M2: Dataset Ready | March 15 | `reviews.csv` | 50 labeled reviews, balanced classes |
| M3: Baseline Model | March 22 | Phase 1 complete | TF-IDF + NB trained, accuracy reported |
| M4: Improved Model | March 29 | Phase 2 complete | 3 models compared, best saved |
| M5: Visualizations | April 05 | R scripts complete | 4 ggplot2 charts generated |
| M6: Web App Ready | April 12 | API + frontend | End-to-end local demo working |
| M7: Deployed | April 15 | Live URL | Accessible on internet |
| M8: Project Delivery | April 19 | Final submission | All docs, code, report submitted |

### 5.4 PERT Chart (Activity Network)

```
                    ┌─────────────┐
                    │ 1.1 Planning │
                    │   (8 hrs)    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ 1.2 Dataset  │
                    │   (4 hrs)    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ 1.3 Phase 1  │
                    │  (12 hrs)    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ 1.4 Phase 2  │
                    │  (16 hrs)    │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌──▼──────────┐ │
       │ 1.5 R Viz   │ │ 1.6 Web App │ │
       │  (8 hrs)    │ │  (12 hrs)   │ │
       └──────┬──────┘ └──────┬──────┘ │
              │               │         │
              └───────┬───────┘         │
                      │                 │
               ┌──────▼──────┐          │
               │ 1.7 Deploy  │          │
               │  (6 hrs)    │          │
               └──────┬──────┘          │
                      │                 │
               ┌──────▼──────┐   ┌─────▼──────┐
               │ 1.8 Testing │   │ 1.9 Docs   │
               │  (8 hrs)    │   │ (10 hrs)    │
               └──────┬──────┘   └──────┬──────┘
                      │                 │
                      └────────┬────────┘
                               │
                        ┌──────▼──────┐
                        │   DELIVERY   │
                        └─────────────┘
```

### 5.5 Critical Path Analysis

**Critical Path:** 1.1 → 1.2 → 1.3 → 1.4 → 1.6 → 1.7 → 1.8 → Delivery

| Activity | Duration | Earliest Start | Earliest Finish | Latest Start | Latest Finish | Slack |
|----------|----------|---------------|-----------------|-------------|---------------|-------|
| 1.1 Planning | 8 hrs | 0 | 8 | 0 | 8 | **0** |
| 1.2 Dataset | 4 hrs | 8 | 12 | 8 | 12 | **0** |
| 1.3 Phase 1 | 12 hrs | 12 | 24 | 12 | 24 | **0** |
| 1.4 Phase 2 | 16 hrs | 24 | 40 | 24 | 40 | **0** |
| 1.5 R Viz | 8 hrs | 40 | 48 | 46 | 54 | 6 hrs |
| 1.6 Web App | 12 hrs | 40 | 52 | 40 | 52 | **0** |
| 1.7 Deploy | 6 hrs | 52 | 58 | 52 | 58 | **0** |
| 1.8 Testing | 8 hrs | 58 | 66 | 58 | 66 | **0** |
| 1.9 Docs | 10 hrs | 58 | 68 | 64 | 74 | 6 hrs |

**Critical Path Duration:** 66 person-hours
**Total Float:** R Visualization (6 hrs slack) and Documentation (6 hrs slack) are non-critical.

---

## 6. PROJECT ESTIMATION AND METRICS

### 6.1 Size Estimation

**Lines of Code (LOC) — Actual:**

| Module | File | LOC (excluding blank lines and comments) |
|--------|------|---:|
| Dataset Generation | `generate_dataset.py` | 42 |
| Phase 1 Baseline | `phase1_basic.py` | 78 |
| Phase 2 Improved | `phase2_improved.py` | 102 |
| Flask API | `app.py` | 82 |
| R Visualization | `visualize_sentiment.R` | 68 |
| R Reticulate | `reticulate_example.R` | 24 |
| Frontend HTML | `index.html` | 72 |
| Frontend CSS | `style.css` | 142 |
| Build Script | `build.py` | 28 |
| **TOTAL** | | **638 LOC** |

### 6.2 Function Point Analysis

| Function Type | Count | Complexity | Weight | FP |
|--------------|------:|-----------|-------:|---:|
| External Inputs (EI) | 2 | Simple | 3 | 6 |
| — Text input via UI | | | | |
| — JSON API request | | | | |
| External Outputs (EO) | 3 | Simple | 4 | 12 |
| — JSON API response | | | | |
| — Web UI result display | | | | |
| — R visualization plots | | | | |
| Internal Logical Files (ILF) | 3 | Simple | 7 | 21 |
| — Training dataset (CSV) | | | | |
| — Trained model files (PKL) | | | | |
| — Prediction results (CSV) | | | | |
| External Interface Files (EIF) | 1 | Simple | 5 | 5 |
| — NLTK stopwords corpus | | | | |
| External Inquiries (EQ) | 1 | Simple | 3 | 3 |
| — GET / (serve frontend) | | | | |
| **Unadjusted Function Points (UFP)** | | | | **47** |

**Value Adjustment Factor (VAF):**

| General System Characteristic | Score (0–5) |
|------------------------------|:-----------:|
| Data communications | 3 |
| Distributed data processing | 0 |
| Performance | 2 |
| Heavily used configuration | 1 |
| Transaction rate | 1 |
| Online data entry | 3 |
| End-user efficiency | 3 |
| Online update | 0 |
| Complex processing | 3 |
| Reusability | 2 |
| Installation ease | 4 |
| Operational ease | 4 |
| Multiple sites | 2 |
| Facilitate change | 2 |
| **Total Degree of Influence (TDI)** | **30** |

```
VAF = 0.65 + (0.01 × TDI) = 0.65 + 0.30 = 0.95
Adjusted FP = UFP × VAF = 47 × 0.95 = 44.65 ≈ 45 FP
```

### 6.3 Effort Estimation (COCOMO — Basic Model)

Using the **Organic** mode (small team, familiar tools):

```
KLOC = 0.638 (638 lines of code)

Effort (person-months) = a × (KLOC)^b
                       = 2.4 × (0.638)^1.05
                       = 2.4 × 0.626
                       = 1.50 person-months

Development Time (months) = c × (Effort)^d
                           = 2.5 × (1.50)^0.38
                           = 2.5 × 1.17
                           = 2.93 months ≈ 3 months

Average Staff = Effort / Time = 1.50 / 2.93 = 0.51 ≈ 1 person
```

**Note:** COCOMO estimates a longer timeline because it assumes industry-standard development with extensive testing and documentation. Our 6-week academic timeline is feasible with a 4-person team working part-time.

### 6.4 Productivity Metrics

| Metric | Value |
|--------|-------|
| Total LOC | 638 |
| Total effort | 84 person-hours |
| Productivity | 638 / 84 = **7.6 LOC/person-hour** |
| Function Points delivered | 45 FP |
| FP productivity | 45 / 84 = **0.54 FP/person-hour** |
| Defect density (target) | < 5 defects per KLOC |
| Actual defects found | 2 (SSL certificate issue, port conflict) |
| Actual defect density | 2 / 0.638 = **3.13 defects/KLOC** |

### 6.5 Software Quality Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Code modularity (files per function area) | ≥ 1 | 9 modules | Met |
| API response time | < 500ms | ~50ms | Exceeded |
| Model accuracy (sample data) | > 50% | 70% (SVM) | Met |
| Model accuracy (IMDB benchmark) | > 85% | 88-90% (published) | Expected |
| Test coverage (manual) | 100% of endpoints | 100% | Met |
| Documentation completeness | README + Report + SEM | All 3 delivered | Met |
| Deployment success | Live URL | Deployed on Render | Met |

---

## 7. FINANCIAL PLANNING AND COST ESTIMATION

### 7.1 Cost Breakdown Structure

| Category | Item | Unit Cost | Quantity | Total Cost |
|----------|------|----------|----------|-----------|
| **Software** | | | | |
| | Python 3.11 | Rs. 0 (open-source) | 1 | Rs. 0 |
| | R 4.0 | Rs. 0 (open-source) | 1 | Rs. 0 |
| | scikit-learn, NLTK, Flask | Rs. 0 (open-source) | 1 | Rs. 0 |
| | ggplot2, dplyr | Rs. 0 (open-source) | 1 | Rs. 0 |
| | VS Code IDE | Rs. 0 (free) | 4 | Rs. 0 |
| | Git + GitHub | Rs. 0 (free for public repos) | 1 | Rs. 0 |
| **Hardware** | | | | |
| | Development laptops | Rs. 0 (existing) | 4 | Rs. 0 |
| | Server/GPU | Rs. 0 (not required) | 0 | Rs. 0 |
| **Cloud & Hosting** | | | | |
| | Render (free tier) | Rs. 0/month | 1 | Rs. 0 |
| | Domain name | Rs. 0 (using Render subdomain) | 1 | Rs. 0 |
| **Data** | | | | |
| | IMDB dataset | Rs. 0 (public) | 1 | Rs. 0 |
| | Sample dataset | Rs. 0 (self-created) | 1 | Rs. 0 |
| **Human Resources** | | | | |
| | Student developers (4) | Rs. 0 (academic project) | 84 hrs | Rs. 0 |
| **Training** | | | | |
| | Online tutorials/documentation | Rs. 0 (free resources) | — | Rs. 0 |
| | | | | |
| **TOTAL PROJECT COST** | | | | **Rs. 0** |

### 7.2 Cost-Benefit Analysis

| Factor | Details |
|--------|---------|
| **Development Cost** | Rs. 0 — entirely open-source and free-tier |
| **Opportunity Cost** | 84 person-hours of student time (equivalent to Rs. 42,000 at Rs. 500/hr industry rate) |
| **Benefits** | Resume-ready project, ML skill development, deployed portfolio piece, interview preparation |
| **ROI** | Intangible but high — directly translatable to job readiness |

### 7.3 Budget for Production Scale-Up (If Required)

| Item | Monthly Cost | Annual Cost |
|------|-------------|-------------|
| Render Pro (always-on, no cold starts) | Rs. 600 | Rs. 7,200 |
| Custom domain (.com) | Rs. 80 | Rs. 960 |
| AWS S3 for large dataset storage | Rs. 100 | Rs. 1,200 |
| **Total (Production)** | **Rs. 780/month** | **Rs. 9,360/year** |

---

## 8. RESOURCE ALLOCATION

### 8.1 Resource Inventory

| Resource Type | Resource | Quantity | Availability |
|--------------|----------|---------|-------------|
| Human | Student developers | 4 | Part-time (15 hrs/week each) |
| Human | Faculty guide | 1 | 2 hrs/week |
| Hardware | Laptops (personal) | 4 | Full-time |
| Software | Python, R, VS Code | — | Free/open-source |
| Infrastructure | GitHub, Render | — | Free tier |

### 8.2 Resource Allocation Matrix (RAM)

| Task | Member 1 (Pulkit) | Member 2 | Member 3 | Member 4 |
|------|:-:|:-:|:-:|:-:|
| Project planning & coordination | **R** | C | C | C |
| Dataset preparation | **R** | A | I | I |
| Phase 1 — TF-IDF + Naive Bayes | A | **R** | I | I |
| Phase 2 — Preprocessing | **R** | A | I | I |
| Phase 2 — Model comparison (LR, SVM) | A | **R** | C | I |
| R Visualization | I | I | **R** | A |
| Flask API development | **R** | I | A | C |
| Frontend (HTML/CSS/JS) | I | I | C | **R** |
| Deployment (Git, Render) | **R** | A | I | I |
| Testing | C | C | **R** | A |
| Documentation (README, Report) | **R** | C | C | C |
| SEM Documentation | **R** | A | C | C |

**Legend:** **R** = Responsible, **A** = Accountable, **C** = Consulted, **I** = Informed

### 8.3 Effort Distribution (Person-Hours)

| Team Member | Role | Hours Allocated | % of Total |
|------------|------|:----:|:----:|
| Member 1 (Pulkit) | Project Lead, ML Engineer, DevOps | 30 | 35.7% |
| Member 2 | ML Engineer, Backend | 22 | 26.2% |
| Member 3 | R Programmer, Tester | 16 | 19.0% |
| Member 4 | Frontend Developer, Tester | 16 | 19.0% |
| **Total** | | **84** | **100%** |

### 8.4 Resource Loading Chart

```
Member 1 (Pulkit — Lead):
W1: ████████░░  (8 hrs — Planning, Dataset)
W2: ████████░░  (8 hrs — Phase 1, Phase 2 preprocessing)
W3: ██████░░░░  (6 hrs — Phase 2 models, API start)
W4: ████░░░░░░  (4 hrs — Deployment)
W5: ██░░░░░░░░  (2 hrs — Testing support)
W6: ██░░░░░░░░  (2 hrs — Documentation)

Member 2 (ML Engineer):
W1: ██░░░░░░░░  (2 hrs — Planning support)
W2: ████████░░  (8 hrs — Phase 1, Phase 2 models)
W3: ██████░░░░  (6 hrs — Model comparison, cross-validation)
W4: ████░░░░░░  (4 hrs — API support)
W5: ██░░░░░░░░  (2 hrs — Testing)
W6: ░░░░░░░░░░  (0 hrs)

Member 3 (R + Testing):
W1: ░░░░░░░░░░  (0 hrs)
W2: ░░░░░░░░░░  (0 hrs)
W3: ██░░░░░░░░  (2 hrs — Learn R/ggplot2)
W4: ██████░░░░  (6 hrs — R visualization scripts)
W5: ██████░░░░  (6 hrs — Testing)
W6: ██░░░░░░░░  (2 hrs — Documentation review)

Member 4 (Frontend):
W1: ░░░░░░░░░░  (0 hrs)
W2: ░░░░░░░░░░  (0 hrs)
W3: ██░░░░░░░░  (2 hrs — UI design)
W4: ██████░░░░  (6 hrs — HTML/CSS/JS development)
W5: ██████░░░░  (6 hrs — UI testing, bug fixes)
W6: ██░░░░░░░░  (2 hrs — Documentation support)
```

---

## 9. TEAM ORGANISATION

### 9.1 Organisation Structure

```
                    ┌─────────────────────┐
                    │    Faculty Guide     │
                    │   (Supervisor)       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Pulkit Raj         │
                    │   Project Lead       │
                    │   ML Engineer        │
                    │   DevOps             │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼─────────┐ ┌───▼───────────┐ ┌──▼──────────────┐
    │   Member 2        │ │   Member 3    │ │   Member 4      │
    │   ML Engineer     │ │   R Programmer│ │   Frontend Dev  │
    │   Backend Dev     │ │   Tester      │ │   UI/UX         │
    └───────────────────┘ └───────────────┘ └─────────────────┘
```

### 9.2 Roles and Responsibilities

| Role | Member | Responsibilities |
|------|--------|-----------------|
| **Project Lead** | Pulkit Raj | Overall coordination, task assignment, progress tracking, stakeholder communication, final integration, deployment, documentation ownership |
| **ML Engineer 1** | Pulkit Raj | Text preprocessing, TF-IDF implementation, model serialization, API development |
| **ML Engineer 2** | Member 2 | Phase 1 baseline model, Phase 2 model comparison (NB, LR, SVM), cross-validation, evaluation metrics |
| **R Programmer / Tester** | Member 3 | R visualization scripts (ggplot2), reticulate integration example, test case design, unit testing, integration testing |
| **Frontend Developer** | Member 4 | HTML structure, CSS styling (dark theme, responsive), JavaScript (fetch API, DOM manipulation), UI/UX testing |
| **Faculty Guide** | Professor | Weekly review, methodology guidance, grading |

### 9.3 Team Communication Protocol

| Channel | Purpose | Frequency |
|---------|---------|-----------|
| WhatsApp Group | Daily updates, quick questions | Daily |
| In-person Meeting | Sprint review, code review, integration | Weekly (1 hr) |
| GitHub Issues | Bug tracking, feature requests | As needed |
| GitHub Pull Requests | Code review before merging | Per feature |
| Email to Faculty | Progress reports, milestone updates | Bi-weekly |

### 9.4 Decision-Making Authority

| Decision Type | Authority | Escalation |
|--------------|-----------|------------|
| Code implementation approach | Individual developer | Project Lead |
| Technology/library choice | Project Lead | Faculty Guide |
| Schedule changes | Project Lead | Faculty Guide |
| Scope changes | Faculty Guide | Department Head |
| Final submission approval | Faculty Guide | — |

---

## 10. RISK MANAGEMENT

### 10.1 Risk Register

| Risk ID | Risk Description | Probability | Impact | Severity | Mitigation Strategy | Contingency Plan |
|---------|-----------------|:-----------:|:------:|:--------:|--------------------|--------------------|
| R-01 | Small dataset leads to low model accuracy | High | Medium | High | Document that accuracy improves with larger datasets; show IMDB benchmarks | Include comparison table with published benchmarks |
| R-02 | Dependency version conflicts during deployment | Medium | High | High | Use version ranges in requirements.txt, not exact pins | Debug with Render build logs; test with multiple Python versions |
| R-03 | Team member unavailable during critical phase | Medium | Medium | Medium | Cross-train members on adjacent modules; keep documentation current | Redistribute tasks; extend timeline by 1 week |
| R-04 | Render free tier cold starts (30s delay) | High | Low | Low | Document behavior in README; acceptable for academic project | Upgrade to paid tier (Rs. 600/month) |
| R-05 | NLTK download fails in CI/CD (SSL issues) | Medium | High | High | Add SSL certificate bypass in build script | Bundle stopwords list directly in repo |
| R-06 | R not installed on team member's machine | Low | Medium | Low | R visualization is optional; provide pre-generated PNGs | Use Google Colab for R execution |
| R-07 | GitHub repository access issues | Low | High | Medium | All members added as collaborators at project start | Use USB/email for code sharing as fallback |
| R-08 | Scope creep (adding deep learning, multi-language support) | Medium | Medium | Medium | Strict scope statement; changes require Faculty Guide approval | Move additions to "Future Scope" section |

### 10.2 Risk Matrix

```
                        IMPACT
                  Low     Medium    High
              ┌─────────┬─────────┬─────────┐
    High      │  R-04   │  R-01   │         │
              ├─────────┼─────────┼─────────┤
PROBABILITY   │         │ R-03    │ R-02    │
    Medium    │         │ R-08    │ R-05    │
              ├─────────┼─────────┼─────────┤
    Low       │  R-06   │ R-07    │         │
              └─────────┴─────────┴─────────┘
```

### 10.3 Risks Encountered and Resolution

| Risk | When | What Happened | How It Was Resolved |
|------|------|---------------|---------------------|
| R-02 | Deployment (Week 5) | `scikit-learn==1.4.2` triggered source build on Render, pulling incompatible `numpy==2.0.0rc1` | Changed from exact version pins to version ranges in `requirements.txt` |
| R-05 | Development (Week 3) | NLTK `stopwords` download failed due to macOS SSL certificate verification | Added `ssl._create_unverified_context` workaround in Python scripts |

---

## 11. QUALITY ASSURANCE PLAN

### 11.1 Quality Standards

| Standard | Application |
|----------|-------------|
| PEP 8 | Python code style (naming conventions, indentation, line length) |
| PEP 257 | Python docstrings for all functions and modules |
| REST API conventions | Proper HTTP methods (GET/POST), status codes (200/400), JSON format |
| Responsive design | CSS works on desktop (1920px) and mobile (375px) |

### 11.2 Test Cases

| TC ID | Test Scenario | Input | Expected Output | Status |
|-------|--------------|-------|-----------------|--------|
| TC-01 | Valid positive review | `{"text": "Amazing product!"}` | `{"sentiment": "positive"}` | Pass |
| TC-02 | Valid negative review | `{"text": "Terrible quality"}` | `{"sentiment": "negative"}` | Pass |
| TC-03 | Empty text input | `{"text": ""}` | `{"error": "Text cannot be empty"}`, HTTP 400 | Pass |
| TC-04 | Missing text field | `{}` | `{"error": "Missing 'text' field"}`, HTTP 400 | Pass |
| TC-05 | Negation handling | `{"text": "Not good at all"}` | `{"sentiment": "negative"}` | Pass |
| TC-06 | Frontend renders correctly | Open http://localhost:5001 | Page loads, textarea visible, button clickable | Pass |
| TC-07 | Example buttons work | Click "This product is absolutely amazing!" | Positive result shown with green styling | Pass |
| TC-08 | Model files exist after training | Run `phase2_improved.py` | `models/best_model.pkl` created | Pass |
| TC-09 | R script executes | Run `Rscript r/visualize_sentiment.R` | 4 PNG files created in `data/` | Pass |
| TC-10 | Deployment build succeeds | Push to GitHub, trigger Render deploy | Build log shows "Build successful" | Pass |

### 11.3 Code Review Checklist

- [ ] Functions have docstrings explaining purpose and parameters
- [ ] No hardcoded file paths (use `os.path.join` and relative references)
- [ ] API validates all input before processing
- [ ] No sensitive data (API keys, passwords) in codebase
- [ ] Model and vectorizer are loaded once at startup, not per request
- [ ] Error messages are user-friendly (no stack traces in API responses)
- [ ] CSS uses responsive units (%, rem) not fixed pixels for layout

---

## 12. CONFIGURATION MANAGEMENT

### 12.1 Version Control

| Item | Detail |
|------|--------|
| VCS Tool | Git |
| Repository | https://github.com/Pulkit2903/sentiment-analyzer |
| Branching Strategy | Single `main` branch (academic project — no feature branching needed) |
| Commit Convention | Descriptive messages with imperative mood ("Add Phase 2 model comparison") |

### 12.2 Configuration Items

| CI ID | Item | Type | Location | Versioned |
|-------|------|------|----------|-----------|
| CI-01 | Python source code | Code | `python/`, `api/` | Yes (Git) |
| CI-02 | R scripts | Code | `r/` | Yes (Git) |
| CI-03 | Frontend files | Code | `templates/`, `static/` | Yes (Git) |
| CI-04 | Trained models | Binary | `models/*.pkl` | Yes (Git) |
| CI-05 | Dataset | Data | `data/reviews.csv` | Yes (Git) |
| CI-06 | Dependencies | Config | `requirements.txt` | Yes (Git) |
| CI-07 | Deployment config | Config | `render.yaml`, `Procfile`, `build.py` | Yes (Git) |
| CI-08 | Documentation | Docs | `README.md`, `REPORT.md`, `SEM_DOCUMENTATION.md` | Yes (Git) |

### 12.3 Release History

| Version | Date | Commit | Description |
|---------|------|--------|-------------|
| v1.0.0 | April 18, 2026 | `7942c96` | Initial release — full pipeline, API, frontend |
| v1.0.1 | April 18, 2026 | `fc93986` | Fix: version ranges in requirements.txt for Render compatibility |
| v1.0.2 | April 18, 2026 | `4562c08` | Add project report (REPORT.md) |

---

## 13. COMMUNICATION PLAN

### 13.1 Stakeholder Communication Matrix

| Stakeholder | Information Needed | Format | Frequency | Responsible |
|-------------|-------------------|--------|-----------|-------------|
| Faculty Guide | Progress, blockers, decisions | In-person meeting | Weekly | Project Lead |
| Team Members | Task assignments, status updates | WhatsApp + GitHub | Daily | Project Lead |
| Faculty Guide | Milestone completion | Email with screenshots | At each milestone | Project Lead |
| Department | Final project submission | Report + code + demo | Once (end of project) | Project Lead |

### 13.2 Meeting Schedule

| Meeting | Day/Time | Duration | Attendees | Agenda |
|---------|----------|----------|-----------|--------|
| Daily Standup | WhatsApp text | 5 min | All 4 members | Yesterday's work, today's plan, blockers |
| Weekly Review | Monday 4 PM | 1 hour | All 4 members | Demo progress, code review, plan next week |
| Faculty Check-in | Wednesday 2 PM | 30 min | Project Lead + Guide | Milestone status, guidance needed |
| Final Demo | April 19 | 1 hour | All + Faculty | Full project demonstration |

### 13.3 Status Report Template

```
PROJECT STATUS REPORT
Date: ___________
Sprint: Week __ of 6

COMPLETED THIS WEEK:
- [Task 1]
- [Task 2]

IN PROGRESS:
- [Task 3] — expected completion: [date]

BLOCKERS:
- [Issue] — action needed: [what]

NEXT WEEK PLAN:
- [Task 4]
- [Task 5]

MILESTONE STATUS: [On Track / At Risk / Delayed]
```

---

## 14. PROJECT CLOSURE

### 14.1 Deliverables Checklist

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | Source code (GitHub repository) | Delivered |
| 2 | Trained ML models (.pkl files) | Delivered |
| 3 | R visualization scripts | Delivered |
| 4 | Flask API + web frontend | Delivered |
| 5 | Live deployed application (Render) | Delivered |
| 6 | Sample dataset (reviews.csv) | Delivered |
| 7 | README.md (setup instructions, tech stack) | Delivered |
| 8 | REPORT.md (problem statement, methodology, results) | Delivered |
| 9 | SEM_DOCUMENTATION.md (this document) | Delivered |

### 14.2 Lessons Learned

| # | Lesson | Category |
|---|--------|----------|
| 1 | Use version ranges (`>=1.3,<2.0`) instead of exact pins in `requirements.txt` — exact pins break on different Python versions and deployment environments | Deployment |
| 2 | Always handle SSL certificate issues when downloading resources in CI/CD — managed environments often have different certificate stores than development machines | Infrastructure |
| 3 | Port 5000 conflicts with macOS AirPlay Receiver — use alternative ports (5001) or disable AirPlay | Development |
| 4 | Small datasets (50 samples) are sufficient for demonstrating a pipeline but not for evaluating model performance — always include benchmark comparisons | ML Best Practice |
| 5 | Keeping negation words during stopword removal is critical for sentiment analysis accuracy | NLP Domain Knowledge |
| 6 | Folder names with trailing spaces cause duplicate directory issues — always validate path names | File System |

### 14.3 Project Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Project Lead | Pulkit Raj | _____________ | __ / __ / 2026 |
| Team Member 2 | _____________ | _____________ | __ / __ / 2026 |
| Team Member 3 | _____________ | _____________ | __ / __ / 2026 |
| Team Member 4 | _____________ | _____________ | __ / __ / 2026 |
| Faculty Guide | _____________ | _____________ | __ / __ / 2026 |

---

*Document prepared as part of the Software Engineering Management coursework.*
*Project Repository: https://github.com/Pulkit2903/sentiment-analyzer*
