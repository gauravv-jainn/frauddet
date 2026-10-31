"""Second verification pass: look every candidate DOI up in Crossref and keep it ONLY if the registry record agrees
with what we cite (title similarity >= 0.90, first-author family name, year within 1). Candidates that fail stay
UNVERIFIED with no DOI. Also stores volume/issue/pages for the IEEE reference list.

Output: tables/references_verified_final.csv. Also prints the abstracts of the arXiv sources used in PROGRESS M11.
"""

from __future__ import annotations

import csv
import difflib
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
UA = "FraudLensReportKit/1.0 (mailto:gaurav.jainn03@gmail.com)"

# key, first author family, year, title as cited, candidate DOI (a CANDIDATE only: it is kept only if Crossref confirms it)
CANDIDATES = [
    ("tanouz2021", "Tanouz", 2021, "Credit Card Fraud Detection Using Machine Learning", "10.1109/ICICCS51141.2021.9432308"),
    ("dalpozzolo2015cal", "Dal Pozzolo", 2015, "Calibrating Probability with Undersampling for Unbalanced Classification", "10.1109/SSCI.2015.33"),
    ("hegarcia2009", "He", 2009, "Learning from Imbalanced Data", "10.1109/TKDE.2008.239"),
    ("dalpozzolo2014", "Dal Pozzolo", 2014, "Learned lessons in credit card fraud detection from a practitioner perspective", "10.1016/j.eswa.2014.02.026"),
    ("bhattacharyya2011", "Bhattacharyya", 2011, "Data mining for credit card fraud: A comparative study", "10.1016/j.dss.2010.08.008"),
    ("randhawa2018", "Randhawa", 2018, "Credit Card Fraud Detection Using AdaBoost and Majority Voting", "10.1109/ACCESS.2018.2806420"),
    ("chawla2002", "Chawla", 2002, "SMOTE: Synthetic Minority Over-sampling Technique", "10.1613/jair.953"),
    ("chen2016", "Chen", 2016, "XGBoost: A Scalable Tree Boosting System", "10.1145/2939672.2939785"),
    ("saito2015", "Saito", 2015, "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets", "10.1371/journal.pone.0118432"),
    ("davis2006", "Davis", 2006, "The relationship between Precision-Recall and ROC curves", "10.1145/1143844.1143874"),
    ("kaufman2012", "Kaufman", 2012, "Leakage in data mining: Formulation, detection, and avoidance", "10.1145/2382577.2382579"),
    ("ribeiro2016", "Ribeiro", 2016, "Why Should I Trust You?: Explaining the Predictions of Any Classifier", "10.1145/2939672.2939778"),
    ("breiman2001", "Breiman", 2001, "Random Forests", "10.1023/A:1010933404324"),
    ("fernandez2018", "Fernandez", 2018, "SMOTE for Learning from Imbalanced Data: Progress and Challenges, Marking the 15-year Anniversary", "10.1613/jair.1.11192"),
    ("akiba2019", "Akiba", 2019, "Optuna: A Next-generation Hyperparameter Optimization Framework", "10.1145/3292500.3330701"),
    ("niculescu2005", "Niculescu-Mizil", 2005, "Predicting good probabilities with supervised learning", "10.1145/1102351.1102430"),
    ("dalpozzolo2018", "Dal Pozzolo", 2018, "Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy", "10.1109/TNNLS.2017.2736643"),
]


def get(url: str) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=40) as r:  # noqa: S310
            return r.status, r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"<[^>]+>", "", s).lower())


rows = []
for key, fam, year, title, doi in CANDIDATES:
    code, body = get(f"https://api.crossref.org/works/{doi}")
    rec = {"key": key, "cited_title": title, "cited_year": year, "candidate_doi": doi, "status": "UNVERIFIED", "doi": "",
           "registry_title": "", "authors": "", "container": "", "volume": "", "issue": "", "pages": "", "year": "", "similarity": "", "note": ""}
    if code == 200:
        m = json.loads(body)["message"]
        t = (m.get("title") or [""])[0]
        full = f"{t}: {(m.get('subtitle') or [''])[0]}" if m.get("subtitle") else t
        s = max(difflib.SequenceMatcher(None, norm(title), norm(x)).ratio() for x in (t, full))
        authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author", [])]
        yr = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
        a0 = (m.get("author") or [{}])[0]
        fa = f"{a0.get('given', '')} {a0.get('family', '')}"  # "Dal Pozzolo" may be split between given and family
        ok = s >= 0.90 and fam.lower() in fa.lower() and yr is not None and abs(yr - year) <= 1
        rec.update(registry_title=full, authors="; ".join(authors), container=(m.get("container-title") or [""])[0], volume=m.get("volume", ""),
                   issue=m.get("issue", ""), pages=m.get("page", m.get("article-number", "")), year=yr, similarity=round(s, 3))
        if ok:
            rec.update(status="VERIFIED", doi=m["DOI"])
        else:
            rec["note"] = "Crossref record for the candidate DOI does not match what we cite"
    else:
        rec["note"] = f"Crossref has no record for the candidate DOI ({body[:60]})"
    rows.append(rec)
    print(key, rec["status"], rec["doi"] or "-", "|", rec["registry_title"][:55], "|", rec["container"][:40], rec["volume"], rec["pages"])
    time.sleep(0.5)

with (KIT / "tables" / "references_verified_final.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# arXiv abstracts used as sources in PROGRESS M11
code, body = get("https://export.arxiv.org/api/query?id_list=1909.01185,2503.20477")
ns = {"a": "http://www.w3.org/2005/Atom"}
for e in ET.fromstring(body).findall("a:entry", ns):
    print("\nARXIV", e.find("a:id", ns).text, "|", re.sub(r"\s+", " ", e.find("a:title", ns).text))
    print(" authors:", "; ".join(a.find("a:name", ns).text for a in e.findall("a:author", ns))[:200], "| published", e.find("a:published", ns).text[:10])
    print(" abstract:", re.sub(r"\s+", " ", e.find("a:summary", ns).text)[:900])
sys.stdout.flush()
