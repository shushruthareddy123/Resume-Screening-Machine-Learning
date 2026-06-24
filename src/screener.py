""
Resume Screening & Ranking System
==================================
Compares TF-IDF + Cosine Similarity vs Sentence-BERT for resume screening.
Dataset: 9,000 resumes | 9 IT job categories

Usage:
    python screener.py                        # runs full pipeline
    python screener.py --jd "your JD text"   # screen with custom JD
"""

import argparse
import re
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# ─── Sample Job Descriptions ─────────────────────────────────────────────────
JOB_DESCRIPTIONS = {
    "Java Developer": """
        Java Developer with 5+ years in J2EE, Spring Boot, Hibernate, REST APIs,
        Maven, SQL databases. Microservices and Agile/Scrum experience required.
    """,
    "Web Developer": """
        Web Developer skilled in HTML5, CSS3, JavaScript, React, Node.js, REST APIs.
        Experience with responsive design, Git. AWS/Azure deployment a plus.
    """,
    "SQL Developer": """
        SQL Developer with strong T-SQL/PL-SQL, stored procedures, indexing,
        performance tuning, ETL, and SSRS. SQL Server or Oracle required.
    """,
    "Business Analyst": """
        Business Analyst for requirements gathering, BRDs, gap analysis, stakeholder
        meetings, process flows. JIRA, Confluence, Agile methodology required.
    """,
    "Network Administrator": """
        Network/Systems Admin with LAN/WAN, firewalls, VPN, TCP/IP, DNS, DHCP.
        CCNA or CompTIA Network+ preferred.
    """,
    "ETL Developer": """
        ETL Developer with Informatica PowerCenter, data warehousing, SSIS, SQL.
        Star/snowflake schema experience preferred.
    """,
    "BI Developer": """
        BI Developer with Tableau, Power BI, Business Objects, SSRS, DAX, SQL.
        Ability to build dashboards and derive business insights.
    """,
    "Project Manager": """
        PMP-certified Project Manager with 7+ years in software delivery, budget
        control, risk management, Agile/Waterfall. MS Project or JIRA required.
    """,
    "Technical Recruiter": """
        Technical Recruiter with full-cycle IT recruiting, LinkedIn sourcing,
        ATS management, and stakeholder coordination skills.
    """,
}

CATEGORY_MAP = {
    "Java Developer":        "Java Developers/Architects Resumes",
    "Web Developer":         "Web Developer Resumes",
    "SQL Developer":         "SQL Developers Resumes",
    "Business Analyst":      "Business Analyst (BA) Resumes",
    "Network Administrator": "Network and Systems Administrators Resumes",
    "ETL Developer":         "Datawarehousing, ETL, Informatica Resumes",
    "BI Developer":          "Business Intelligence, Business Object Resumes",
    "Project Manager":       "Project Manager Resumes",
    "Technical Recruiter":   "Recruiter Resumes",
}


# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── TF-IDF Screening ─────────────────────────────────────────────────────────
class TFIDFScreener:
    def __init__(self, df: pd.DataFrame):
        print("Building TF-IDF matrix...")
        t = time.time()
        self.df = df
        self.vectorizer = TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2),
            stop_words="english", sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(df["clean_text"])
        print(f"  ✅ Matrix: {self.matrix.shape}  ({time.time()-t:.1f}s)")

    def screen(self, job_description: str, top_n: int = 10) -> pd.DataFrame:
        clean_jd = preprocess_text(job_description)
        jd_vec = self.vectorizer.transform([clean_jd])
        scores = cosine_similarity(jd_vec, self.matrix).flatten()
        idx = scores.argsort()[::-1][:top_n]
        result = self.df.iloc[idx][["category", "job_title"]].copy()
        result["score"] = scores[idx].round(4)
        result["rank"] = range(1, top_n + 1)
        return result.reset_index(drop=True)

    def precision_at_k(self, k: int = 10) -> dict:
        scores = {}
        for title, jd in JOB_DESCRIPTIONS.items():
            expected = CATEGORY_MAP[title]
            top = self.screen(jd, k)
            scores[title] = (top["category"] == expected).sum() / k
        return scores

    def train_classifier(self, y):
        X_train, X_test, y_train, y_test = train_test_split(
            self.matrix, y, test_size=0.2, random_state=42, stratify=y
        )
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                                 multi_class="multinomial")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return accuracy_score(y_test, y_pred), y_test, y_pred


# ─── SBERT Screening ──────────────────────────────────────────────────────────
class SBERTScreener:
    def __init__(self, df: pd.DataFrame):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")

        print("Loading Sentence-BERT (all-MiniLM-L6-v2)...")
        self.df = df
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Encoding resumes...")
        t = time.time()
        texts = df["clean_text"].str.split().str[:256].str.join(" ").tolist()
        self.embeddings = self.model.encode(
            texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
        )
        print(f"  ✅ Embeddings: {self.embeddings.shape}  ({time.time()-t:.1f}s)")

    def screen(self, job_description: str, top_n: int = 10) -> pd.DataFrame:
        jd_vec = self.model.encode([preprocess_text(job_description)])
        scores = cosine_similarity(jd_vec, self.embeddings).flatten()
        idx = scores.argsort()[::-1][:top_n]
        result = self.df.iloc[idx][["category", "job_title"]].copy()
        result["score"] = scores[idx].round(4)
        result["rank"] = range(1, top_n + 1)
        return result.reset_index(drop=True)

    def precision_at_k(self, k: int = 10) -> dict:
        scores = {}
        for title, jd in JOB_DESCRIPTIONS.items():
            expected = CATEGORY_MAP[title]
            top = self.screen(jd, k)
            scores[title] = (top["category"] == expected).sum() / k
        return scores

    def train_classifier(self, y):
        X_train, X_test, y_train, y_test = train_test_split(
            self.embeddings, y, test_size=0.2, random_state=42, stratify=y
        )
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return accuracy_score(y_test, y_pred), y_test, y_pred


# ─── Main Pipeline ────────────────────────────────────────────────────────────
def run_pipeline(data_path: str = "data.csv", custom_jd: str = None):
    print("\n" + "=" * 65)
    print("       RESUME SCREENING & RANKING SYSTEM")
    print("=" * 65)

    # Load data
    print("\n[1/5] Loading dataset...")
    df = pd.read_csv(data_path)
    df["clean_text"] = df["Text"].apply(preprocess_text)
    print(f"  ✅ {len(df)} resumes | {df['category'].nunique()} categories")

    le = LabelEncoder()
    y = le.fit_transform(df["category"])

    # TF-IDF
    print("\n[2/5] TF-IDF Approach")
    tfidf_screener = TFIDFScreener(df)

    print("\n[3/5] Sentence-BERT Approach")
    sbert_screener = SBERTScreener(df)

    # Precision@10
    print("\n[4/5] Evaluating Precision@10...")
    tfidf_p = tfidf_screener.precision_at_k(10)
    sbert_p  = sbert_screener.precision_at_k(10)

    print(f"\n{'Role':<28} {'TF-IDF':>8} {'SBERT':>8} {'Winner':>8}")
    print("-" * 55)
    for role in tfidf_p:
        t = tfidf_p[role]
        s = sbert_p[role]
        winner = "SBERT" if s > t else ("TF-IDF" if t > s else "Tie")
        print(f"{role:<28} {t:>7.0%} {s:>8.0%} {winner:>8}")
    print("-" * 55)
    print(f"{'Average':<28} {np.mean(list(tfidf_p.values())):>7.0%} "
          f"{np.mean(list(sbert_p.values())):>8.0%}")

    # Classification
    print("\n[5/5] Classification Accuracy (LogReg on features)...")
    acc_t, _, _ = tfidf_screener.train_classifier(y)
    acc_s, _, _ = sbert_screener.train_classifier(y)
    print(f"  TF-IDF + LogReg: {acc_t:.2%}")
    print(f"  SBERT  + LogReg: {acc_s:.2%}")

    # Custom JD demo
    if custom_jd:
        print("\n" + "=" * 65)
        print("SCREENING YOUR CUSTOM JOB DESCRIPTION")
        print("=" * 65)
        print(f"\n🔵 TF-IDF Top 5:")
        print(tfidf_screener.screen(custom_jd, 5).to_string(index=False))
        print(f"\n🟠 SBERT Top 5:")
        print(sbert_screener.screen(custom_jd, 5).to_string(index=False))

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.csv", help="Path to CSV dataset")
    parser.add_argument("--jd", default=None, help="Custom job description text")
    args = parser.parse_args()
    run_pipeline(data_path=args.data, custom_jd=args.jd)
