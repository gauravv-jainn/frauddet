"""FraudLens scoring API.

    set FRAUDLENS_API_KEY=<your key>
    uvicorn api.main:app --port 8000          (run from the fraudlens/ folder)

POST /score with header ``X-API-Key`` and a JSON body of one or more transactions
(same columns as the Kaggle schema). Returns probability, band, action and reasons.
"""
from __future__ import annotations

import os
import secrets
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraudkit.data import clean, validate_schema  # noqa: E402
from fraudkit.explain import explain_frame  # noqa: E402
from fraudkit.features import featurize_with_history  # noqa: E402
from fraudkit.score import ARTIFACTS, FraudModel  # noqa: E402

app = FastAPI(title="FraudLens scoring API", version="1.0")


class Transaction(BaseModel):
    trans_date_trans_time: datetime
    cc_num: int
    merchant: str
    category: str
    amt: float = Field(ge=0)
    gender: str
    city: str
    state: str
    lat: float = Field(ge=-90, le=90)
    long: float = Field(ge=-180, le=180)
    city_pop: int = Field(ge=0)
    job: str
    dob: datetime
    merch_lat: float = Field(ge=-90, le=90)
    merch_long: float = Field(ge=-180, le=180)
    trans_num: str | None = None


class ScoreRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class ReasonOut(BaseModel):
    text: str
    direction: str
    contribution: float


class ScoreOut(BaseModel):
    trans_num: str | None
    probability: float
    band: str
    flagged: bool
    action: str
    summary: str
    reasons: list[ReasonOut]


@lru_cache(maxsize=1)
def model() -> FraudModel:
    return FraudModel.load(ARTIFACTS / "model.joblib")


@lru_cache(maxsize=1)
def history() -> pd.DataFrame:
    p = ARTIFACTS / "history.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def require_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("FRAUDLENS_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="Server has no FRAUDLENS_API_KEY configured")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": (ARTIFACTS / "model.joblib").exists()}


@app.post("/score", response_model=list[ScoreOut], dependencies=[Depends(require_key)])
def score(req: ScoreRequest):
    df = pd.DataFrame([t.model_dump() for t in req.transactions])
    df["trans_num"] = [t if t else f"req_{i}" for i, t in enumerate(df["trans_num"])]
    problems = validate_schema(df)
    if problems:
        raise HTTPException(status_code=422, detail=problems)
    df = clean(df)
    fm = model()
    hist = history()
    if len(hist):
        hist = hist[hist["cc_num"].isin(df["cc_num"].unique()) & ~hist["trans_num"].isin(df["trans_num"])]
    feat = featurize_with_history(df, hist)
    out = []
    for tn, e in zip(df["trans_num"], explain_frame(fm, feat)):
        out.append(ScoreOut(
            trans_num=tn, probability=e.probability, band=e.band, flagged=e.probability >= fm.threshold,
            action=e.action, summary=e.summary,
            reasons=[ReasonOut(text=r.text, direction=r.direction, contribution=r.contribution)
                     for r in e.raising + e.lowering]))
    return out
