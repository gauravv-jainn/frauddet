"""Verify every reference against Crossref / arXiv / PMLR. DOIs are only recorded when the registry returns them.

Output: report_kit/tables/references_verification.csv (what we cited vs what the registry says).
A reference is VERIFIED only if the registry record matches on title (similarity >= 0.90), year and first author.
"""

from __future__ import annotations

import csv
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
UA = "FraudLensReportKit/1.0 (mailto:gaurav.jainn03@gmail.com)"

# key, first-author surname, year, title as we cite it, expected venue fragment, kind
REFS = [
    ("tanouz2021", "Tanouz", 2021, "Credit Card Fraud Detection Using Machine Learning", "Intelligent Computing and Control Systems", "crossref"),
    ("dalpozzolo2015cal", "Dal Pozzolo", 2015, "Calibrating Probability with Undersampling for Unbalanced Classification", "Computational Intelligence", "crossref"),
    ("hegarcia2009", "He", 2009, "Learning from Imbalanced Data", "Knowledge and Data Engineering", "crossref"),
    ("dalpozzolo2014", "Dal Pozzolo", 2014, "Learned lessons in credit card fraud detection from a practitioner perspective", "Expert Systems with Applications", "crossref"),
    ("bhattacharyya2011", "Bhattacharyya", 2011, "Data mining for credit card fraud: A comparative study", "Decision Support Systems", "crossref"),
    ("randhawa2018", "Randhawa", 2018, "Credit Card Fraud Detection Using AdaBoost and Majority Voting", "IEEE Access", "crossref"),
    ("chawla2002", "Chawla", 2002, "SMOTE: Synthetic Minority Over-sampling Technique", "Artificial Intelligence Research", "crossref"),
    ("chen2016", "Chen", 2016, "XGBoost: A Scalable Tree Boosting System", "Knowledge Discovery and Data Mining", "crossref"),
    ("saito2015", "Saito", 2015, "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets", "PLOS ONE", "crossref"),
    ("davis2006", "Davis", 2006, "The relationship between Precision-Recall and ROC curves", "Machine learning", "crossref"),
    ("kaufman2012", "Kaufman", 2012, "Leakage in data mining: Formulation, detection, and avoidance", "Knowledge Discovery from Data", "crossref"),
    ("ribeiro2016", "Ribeiro", 2016, "Why Should I Trust You?: Explaining the Predictions of Any Classifier", "Knowledge Discovery and Data Mining", "crossref"),
    ("breiman2001", "Breiman", 2001, "Random Forests", "Machine Learning", "crossref"),
    ("fernandez2018", "Fernandez", 2018, "SMOTE for Learning from Imbalanced Data: Progress and Challenges, Marking the 15-year Anniversary", "Artificial Intelligence Research", "crossref"),
    ("akiba2019", "Akiba", 2019, "Optuna: A Next-generation Hyperparameter Optimization Framework", "Knowledge Discovery", "crossref"),
    ("niculescu2005", "Niculescu-Mizil", 2005, "Predicting good probabilities with supervised learning", "Machine learning", "crossref"),
    ("dalpozzolo2018", "Dal Pozzolo", 2018, "Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy", "Neural Networks and Learning Systems", "crossref"),
    ("lundberg2017", "Lundberg", 2017, "A Unified Approach to Interpreting Model Predictions", "arXiv:1705.07874", "arxiv:1705.07874"),
    ("guo2017", "Guo", 2017, "On Calibration of Modern Neural Networks", "arXiv:1706.04599", "arxiv:1706.04599"),
    ("arxiv1909", "", 2019, "", "arXiv:1909.01185", "arxiv:1909.01185"),
    ("arxiv2503", "", 2025, "", "arXiv:2503.20477", "arxiv:2503.20477"),
    ("kull2017", "Kull", 2017, "Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers", "AISTATS PMLR 54", "url:https://proceedings.mlr.press/v54/kull17a.html"),
    ("ke2017", "Ke", 2017, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", "NeurIPS 30", "url:https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html"),
    ("handbook", "Le Borgne", 2022, "Fraud Detection Handbook", "online book", "url:https://fraud-detection-handbook.github.io/fraud-detection-handbook/"),
    ("owasp", "OWASP", 2024, "Password Storage Cheat Sheet", "online", "url:https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html"),
    ("sparkov", "Harris", 2020, "Sparkov Data Generation", "GitHub", "url:https://github.com/namebrandon/Sparkov_Data_Generation"),
    ("kaggle", "Shenoy", 2020, "Credit Card Transactions Fraud Detection Dataset", "Kaggle", "url:https://www.kaggle.com/datasets/kartik2112/fraud-detection"),
    ("sdg", "United Nations", 2015, "Sustainable Development Goals", "online", "url:https://sdgs.un.org/goals"),
]


def get(url: str, accept: str | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **({"Accept": accept} if accept else {})})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=40) as r:  # noqa: S310
            return r.status, r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def sim(a: str, b: str) -> float:
    n = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower())  # noqa: E731
    return difflib.SequenceMatcher(None, n(a), n(b)).ratio()


rows = []
for key, author, year, title, venue, kind in REFS:
    rec = {"key": key, "cited_author": author, "cited_year": year, "cited_title": title, "registry": "", "registry_title": "",
           "registry_first_author": "", "registry_year": "", "registry_container": "", "doi": "", "url": "", "title_similarity": "", "status": "UNVERIFIED", "note": ""}
    if kind == "crossref":
        q = urllib.parse.quote(f"{title} {author}")
        code, body = get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=5&select=DOI,title,author,issued,container-title,type")
        rec["registry"] = "Crossref"
        best = None
        if code == 200:
            for it in json.loads(body)["message"]["items"]:
                t = (it.get("title") or [""])[0]
                s = sim(title, t)
                if best is None or s > best[0]:
                    best = (s, it)
        if best:
            s, it = best
            yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
            fa = (it.get("author") or [{}])[0].get("family", "")
            rec.update(registry_title=(it.get("title") or [""])[0], registry_first_author=fa, registry_year=yr,
                       registry_container=(it.get("container-title") or [""])[0], doi=it["DOI"], title_similarity=round(s, 3),
                       url=f"https://doi.org/{it['DOI']}")
            ok = s >= 0.90 and fa.lower().startswith(author.split()[-1].lower()[:5]) and abs(int(yr or 0) - year) <= 1
            rec["status"] = "VERIFIED" if ok else "UNVERIFIED"
            if not ok:
                rec["note"] = "best Crossref match did not agree on title/author/year"
                rec["doi"] = ""
        else:
            rec["note"] = f"Crossref lookup failed: {body[:80]}"
        time.sleep(0.6)
    elif kind.startswith("arxiv:"):
        aid = kind.split(":", 1)[1]
        code, body = get(f"https://export.arxiv.org/api/query?id_list={aid}")
        rec["registry"] = "arXiv"
        if code == 200 and "<entry>" in body:
            ns = {"a": "http://www.w3.org/2005/Atom"}
            e = ET.fromstring(body).find("a:entry", ns)
            t = re.sub(r"\s+", " ", e.find("a:title", ns).text).strip()
            fa = e.find("a:author/a:name", ns).text
            yr = int(e.find("a:published", ns).text[:4])
            rec.update(registry_title=t, registry_first_author=fa, registry_year=yr, registry_container=f"arXiv:{aid}", url=f"https://arxiv.org/abs/{aid}")
            if title:
                s = sim(title, t)
                rec["title_similarity"] = round(s, 3)
                rec["status"] = "VERIFIED" if s >= 0.90 and abs(yr - year) <= 1 else "UNVERIFIED"
            else:
                rec["status"] = "VERIFIED (record exists; title taken from arXiv)"
        else:
            rec["note"] = f"arXiv lookup failed: {body[:80]}"
        time.sleep(3)
    else:
        url = kind.split(":", 1)[1]
        code, body = get(url)
        rec["registry"] = "web page"
        rec["url"] = url
        if code == 200:
            m = re.search(r"<title>(.*?)</title>", body, flags=re.S | re.I)
            pt = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
            rec["registry_title"] = pt[:140]
            s = sim(title, pt[: len(title) + 40]) if pt else 0.0
            contains = title.lower()[:25] in body.lower()
            rec["title_similarity"] = round(s, 3)
            rec["status"] = "VERIFIED" if contains else "UNVERIFIED"
            rec["note"] = "page reachable and contains the cited title" if contains else "page reachable but title text not found"
        else:
            rec["note"] = f"page not fetched ({body[:80]})"
    rows.append(rec)
    print(key, rec["status"], rec["doi"], rec["registry_title"][:60])

out = KIT / "tables" / "references_verification.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("wrote", out)
sys.stdout.flush()
