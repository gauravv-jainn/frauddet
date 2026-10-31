"""Shared loaders and markdown helpers for the REPORT_FACTS.md generator. Every number comes from an artifact,
a table in report_kit/tables, or a command run by these scripts."""

from __future__ import annotations

import ast
import html
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

KIT = Path(__file__).resolve().parents[1]
REPO = KIT.parent
FL = REPO / "fraudlens"
T, D = KIT / "tables", KIT / "_data"

M = json.loads((FL / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
X = json.loads((D / "extra_facts.json").read_text(encoding="utf-8"))
S = json.loads((D / "summary.json").read_text(encoding="utf-8"))
SHAP = json.loads((D / "shap_examples.json").read_text(encoding="utf-8"))
TEST, AT = M["test"], M["test"]["at_threshold"]


def tab(name: str) -> pd.DataFrame:
    """Read a table from report_kit/tables. 'N/A' stays text (pandas would turn it into NaN); HTML entities from Crossref are decoded."""
    df = pd.read_csv(T / name, keep_default_na=False, na_values=[""])
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
            df[c] = df[c].map(lambda v: re.sub(r"\s{2,}", " ", html.unescape(v)) if isinstance(v, str) else v)
    return df


def esc(v) -> str:
    return str(v).replace("|", "\\|").replace("\n", " ")


def md_table(df: pd.DataFrame, cols: list[str] | None = None, heads: list[str] | None = None, fmt: dict | None = None) -> str:
    cols = cols or list(df.columns)
    heads = heads or cols
    fmt = fmt or {}
    out = ["| " + " | ".join(heads) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if pd.isna(v):
                cells.append("")
            elif c in fmt:
                cells.append(esc(fmt[c](v)))
            else:
                cells.append(esc(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def src(*items: str) -> str:
    return "*Source:* " + "; ".join(items) + "\n"


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=True).stdout.strip()  # noqa: S603


def py_files() -> list[str]:
    return [f for f in run(["git", "ls-files", "fraudlens/*.py"]).splitlines() if f.endswith(".py")]


def loc(files: list[str]) -> tuple[int, int]:
    """(physical lines, code lines = non-blank and not starting with #)."""
    phys = code = 0
    for f in files:
        for line in (REPO / f).read_text(encoding="utf-8").splitlines():
            phys += 1
            s = line.strip()
            if s and not s.startswith("#"):
                code += 1
    return phys, code


def docstring_first_line(path: Path) -> str:
    d = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    return (d or "").split("\n")[0]


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def n(x, nd: int = 0) -> str:
    return f"{x:,.{nd}f}"


def pytest_summary() -> dict:
    txt = (D / "pytest_full_run.txt").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"(\d+) failed, (\d+) passed", txt)
    return {"failed": int(m.group(1)), "passed": int(m.group(2))} if m else {}


def card_overlap() -> tuple[int, int, int]:
    """(train cards, test cards, test cards that also appear in train) from the cached Kaggle CSVs."""
    import glob

    base = Path.home() / ".cache" / "kagglehub" / "datasets" / "kartik2112" / "fraud-detection" / "versions"
    tr = glob.glob(str(base / "*" / "fraudTrain.csv"))[0]
    a = set(pd.read_csv(tr, usecols=["cc_num"])["cc_num"])
    b = set(pd.read_csv(tr.replace("fraudTrain", "fraudTest"), usecols=["cc_num"])["cc_num"])
    return len(a), len(b), len(a & b)


def initials(given: str) -> str:
    out = []
    for tok in given.replace(".", " ").split():
        out.append("-".join(f"{p[0]}." for p in tok.split("-") if p))
    return " ".join(out)


def ieee_authors(authors: str) -> str:
    """'given family; given family' (as stored from Crossref) -> IEEE style 'G. Family, G. Family, and G. Family'."""
    names = []
    for a in authors.split(";"):
        parts = a.strip().split()
        fam = " ".join(parts[-2:]) if len(parts) >= 3 and parts[-2].lower() in {"dal", "van", "de", "von", "le"} else parts[-1]
        given = " ".join(parts[: len(parts) - len(fam.split())])
        names.append(f"{initials(given)} {fam}".strip())
    if len(names) > 6:
        return names[0] + " et al."
    if len(names) == 1:
        return names[0]
    return (", ".join(names[:-1]) + (", and " if len(names) > 2 else " and ") + names[-1])
