#  Resume Screening & Ranking System

A complete ML project that screens and ranks resumes against job descriptions using two approaches:
- TF-IDF + Cosine Similarity - fast, keyword driven
- Sentence-BERT** - semantic, context aware

---

## Project Structure

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



