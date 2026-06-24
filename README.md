# 🔍 Resume Screening & Ranking System

A complete ML project that screens and ranks resumes against job descriptions using **two approaches**:
- **TF-IDF + Cosine Similarity** — fast, keyword-driven
- **Sentence-BERT** — semantic, context-aware

---

## 📁 Project Structure

```
resume_screening_project/
│
├── data.csv                        ← Your dataset (9,000 resumes, 9 categories)
├── requirements.txt
├── README.md
│
├── Resume_Screening_ML.ipynb       ← Full Jupyter Notebook (EDA → Results)
│
└── src/
    └── screener.py                 ← Standalone Python pipeline script
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
```bash
jupyter notebook Resume_Screening_ML.ipynb
```
Walk through 11 cells covering EDA, preprocessing, both approaches, and comparison.

### 3. Run the Python script
```bash
# Full evaluation pipeline
python src/screener.py

# Screen with your own job description
python src/screener.py --jd "Looking for a Python Data Engineer with Spark, Airflow, and AWS experience."
```

---

## 📊 Dataset

| Feature     | Details                        |
|-------------|--------------------------------|
| Records     | 9,000 resumes                  |
| Columns     | `category`, `job_title`, `Text`|
| Categories  | 9 IT job domains               |

**Categories:**
- Java Developers/Architects
- Web Developer
- SQL Developers
- Business Analyst (BA)
- Network and Systems Administrators
- Datawarehousing, ETL, Informatica
- Business Intelligence, Business Object
- Project Manager
- Recruiter

---

## 🧠 ML Approaches

### Approach 1 — TF-IDF + Cosine Similarity
- Vectorizes resume text using `TfidfVectorizer` (10K features, bigrams)
- Encodes the job description with the same vectorizer
- Ranks resumes by cosine similarity score
- Also trains a **Logistic Regression** classifier for category prediction

### Approach 2 — Sentence-BERT
- Uses `all-MiniLM-L6-v2` from `sentence-transformers`
- Encodes resumes and JDs into 384-dim dense embeddings
- Ranks by cosine similarity in embedding space
- Also trains a **Logistic Regression** classifier on top of embeddings

---

## 📈 Evaluation Metric

**Precision@10**: For each JD, what fraction of the top-10 ranked resumes belong to the expected category?

---

## 💡 Key Takeaways

| | TF-IDF | SBERT |
|---|---|---|
| Speed | ⚡ Very fast | 🐢 Slower (GPU helps) |
| Keyword matching | ✅ Excellent | ✅ Good |
| Semantic understanding | ❌ Limited | ✅ Excellent |
| Best for | Tech-specific JDs | General/hybrid JDs |

**Recommendation:** Use both in an ensemble for production systems.
