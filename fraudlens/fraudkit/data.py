"""Data loading for FraudLens.

Primary source: Kaggle "kartik2112/fraud-detection" (simulated card transactions,
generated with the Sparkov tool). The provided fraudTrain/fraudTest split is a
time split and is kept as-is.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

KAGGLE_DATASET = "kartik2112/fraud-detection"
DATA_SOURCE = "Kaggle kartik2112/fraud-detection (Sparkov simulated card transactions)"

RAW_COLUMNS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "gender",
    "city", "state", "lat", "long", "city_pop", "job", "dob", "merch_lat",
    "merch_long", "trans_num", "is_fraud",
]
# Columns a scoring request must provide (label and trans_num optional).
REQUIRED_COLUMNS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "gender",
    "city", "state", "lat", "long", "city_pop", "job", "dob", "merch_lat", "merch_long",
]
DTYPES = {
    "cc_num": "int64", "merchant": "string", "category": "string", "amt": "float64",
    "gender": "string", "city": "string", "state": "string", "lat": "float64",
    "long": "float64", "city_pop": "int64", "job": "string", "merch_lat": "float64",
    "merch_long": "float64", "trans_num": "string",
}


@dataclass
class RawData:
    train: pd.DataFrame
    test: pd.DataFrame
    source: str


def locate_dataset(cache_dir: str | os.PathLike | None = None) -> Path:
    """Return the folder containing fraudTrain.csv / fraudTest.csv, downloading if needed."""
    if cache_dir:
        p = Path(cache_dir)
        if (p / "fraudTrain.csv").exists() and (p / "fraudTest.csv").exists():
            return p
    import kagglehub  # imported lazily so the app does not need network access

    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Names, street and zip are not used by the model; drop them so they never reach the app.
    pii = ["first", "last", "street", "zip", "unix_time"]
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c in pii], errors="ignore")
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    for col, dt in DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dt)
    if "merchant" in df.columns:
        # Sparkov prefixes every merchant with "fraud_"; it carries no signal, strip it for display.
        df["merchant"] = df["merchant"].str.replace(r"^fraud_", "", regex=True)
    if "is_fraud" in df.columns:
        df["is_fraud"] = df["is_fraud"].astype("int8")
    return df


def load_raw(cache_dir: str | None = None, sample_cards: int | None = None, seed: int = 42) -> RawData:
    """Load train and test. ``sample_cards`` keeps a random subset of cards (used by --fast)."""
    folder = locate_dataset(cache_dir)
    usecols = lambda c: not c.startswith("Unnamed")  # noqa: E731
    train = clean(pd.read_csv(folder / "fraudTrain.csv", usecols=usecols))
    test = clean(pd.read_csv(folder / "fraudTest.csv", usecols=usecols))
    if sample_cards:
        import numpy as np

        cards = np.sort(train["cc_num"].unique())
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(cards, size=min(sample_cards, len(cards)), replace=False).tolist())
        train = train[train["cc_num"].isin(keep)].reset_index(drop=True)
        test = test[test["cc_num"].isin(keep)].reset_index(drop=True)
    return RawData(train=train, test=test, source=DATA_SOURCE)


def describe(df: pd.DataFrame) -> dict:
    t = df["trans_date_trans_time"]
    return {
        "rows": int(len(df)),
        "cards": int(df["cc_num"].nunique()),
        "frauds": int(df["is_fraud"].sum()),
        "fraud_rate": float(df["is_fraud"].mean()),
        "start": str(t.min()),
        "end": str(t.max()),
    }


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return human-readable problems with an uploaded frame (empty list = OK)."""
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append("Missing required columns: " + ", ".join(missing))
        return problems
    if len(df) == 0:
        problems.append("The file has no rows.")
        return problems
    for col in ["trans_date_trans_time", "dob"]:
        bad = pd.to_datetime(df[col], errors="coerce").isna().sum()
        if bad:
            problems.append(f"Column '{col}' has {bad} value(s) that are not valid dates.")
    for col in ["amt", "lat", "long", "merch_lat", "merch_long", "city_pop", "cc_num"]:
        bad = pd.to_numeric(df[col], errors="coerce").isna().sum()
        if bad:
            problems.append(f"Column '{col}' has {bad} non-numeric or empty value(s).")
    if not problems:
        amt = pd.to_numeric(df["amt"], errors="coerce")
        if (amt < 0).any():
            problems.append("Column 'amt' has negative amounts.")
        lat = pd.to_numeric(df["lat"], errors="coerce")
        if ((lat < -90) | (lat > 90)).any():
            problems.append("Column 'lat' has values outside -90..90.")
    return problems


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    con = Console()
    raw = load_raw()
    con.print(f"[bold]Source:[/] {raw.source}")
    con.print(raw.train.dtypes.to_string())
    tbl = Table("split", "rows", "cards", "frauds", "fraud_rate", "start", "end")
    for name, df in [("train", raw.train), ("test", raw.test)]:
        d = describe(df)
        tbl.add_row(name, f"{d['rows']:,}", f"{d['cards']:,}", f"{d['frauds']:,}",
                    f"{d['fraud_rate']:.4%}", d["start"], d["end"])
    con.print(tbl)
