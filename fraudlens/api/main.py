"""FraudLens scoring API.

    uvicorn api.main:app --port 8000          (run from the fraudlens/ folder)

Authentication: header ``X-API-Key``. Keys are created by an admin in the app (Admin -> API keys)
or with ``python -m fraudkit.admin create-api-key``; only their hash is stored. For a single
bootstrap key the environment variable FRAUDLENS_API_KEY is also accepted.

    POST /v1/score   score 1..N transactions -> probability, band, action, reasons, model version
    GET  /v1/model   served model version, threshold and headline test metrics
    GET  /v1/health  liveness            GET /v1/ready   model loaded and database reachable

Every response carries an X-Request-ID header; logs are one JSON line per request and never
contain card numbers.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraudkit.auth import ApiKeyStore
from fraudkit.config import load_config
from fraudkit.data import clean, validate_schema
from fraudkit.db import Database
from fraudkit.explain import explain_frame
from fraudkit.features import featurize_with_history
from fraudkit.score import ARTIFACTS, FraudModel, load_metrics, load_serving_model

log = logging.getLogger("fraudlens.api")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)

app = FastAPI(
    title="FraudLens scoring API", version="1.1", description="Calibrated card-fraud scores with plain-English reasons."
)


# --------------------------------------------------------------------------- schemas
class Transaction(BaseModel):
    trans_date_trans_time: datetime
    cc_num: int
    merchant: str = Field(max_length=200)
    category: str = Field(max_length=100)
    amt: float = Field(ge=0, le=10_000_000)
    gender: str = Field(max_length=20)
    city: str = Field(max_length=200)
    state: str = Field(max_length=50)
    lat: float = Field(ge=-90, le=90)
    long: float = Field(ge=-180, le=180)
    city_pop: int = Field(ge=0)
    job: str = Field(max_length=200)
    dob: datetime
    merch_lat: float = Field(ge=-90, le=90)
    merch_long: float = Field(ge=-180, le=180)
    trans_num: str | None = Field(default=None, max_length=100)


class ScoreRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1)


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


class ScoreResponse(BaseModel):
    model_version: str
    threshold: float
    request_id: str
    results: list[ScoreOut]


# --------------------------------------------------------------------------- dependencies
@lru_cache(maxsize=1)
def settings():
    return load_config()


@lru_cache(maxsize=1)
def database() -> Database:
    return Database(settings().db_path())


_model_cache: dict = {}
_model_lock = threading.Lock()


def model() -> FraudModel:
    """Reload automatically when a retrain replaces model.joblib (atomic swap by the trainer)."""
    path = ARTIFACTS / "model.joblib"
    mtime = path.stat().st_mtime
    with _model_lock:
        if _model_cache.get("mtime") != mtime:
            _model_cache.update(model=load_serving_model(path, bands=settings().bands.model_dump()), mtime=mtime)
            history.cache_clear()
        return _model_cache["model"]


@lru_cache(maxsize=1)
def history() -> pd.DataFrame:
    p = ARTIFACTS / "history.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


class RateLimiter:
    """Sliding one-minute window per client (in-process; use a gateway/Redis when running several workers)."""

    def __init__(self):
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, client: str, limit: int) -> int | None:
        """Returns seconds to wait if the client is over the limit, else None."""
        now = time.monotonic()
        with self.lock:
            q = self.hits[client]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= limit:
                return int(60 - (now - q[0])) + 1
            q.append(now)
            return None


limiter = RateLimiter()


def authenticate(request: Request, x_api_key: str | None = Header(default=None)) -> str:
    """Returns the client name. DB keys first, then the optional bootstrap env key."""
    rec = ApiKeyStore(database()).verify(x_api_key)
    client = None
    if rec is not None:
        client = f"{rec.key_id}:{rec.name}"
    else:
        env_key = os.environ.get("FRAUDLENS_API_KEY")
        if env_key and x_api_key and secrets.compare_digest(x_api_key, env_key):
            client = "env-key"
    if client is None:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")
    wait = limiter.check(client, settings().api.rate_limit_per_minute)
    if wait is not None:
        raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"Retry-After": str(wait)})
    request.state.client = client
    return client


# --------------------------------------------------------------------------- middleware
@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    rid = "".join(ch for ch in rid if ch.isalnum() or ch in "-_")[:64] or uuid.uuid4().hex
    request.state.request_id = rid
    request.state.client = None
    request.state.n_txn = None
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # never leak a stack trace to the client
        log.exception(json.dumps({"event": "unhandled_error", "request_id": rid, "path": request.url.path}))
        response = JSONResponse(status_code=500, content={"detail": "Internal error", "request_id": rid})
    response.headers["X-Request-ID"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    log.info(
        json.dumps(
            {
                "event": "request",
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round((time.perf_counter() - t0) * 1000, 1),
                "client": request.state.client,
                "transactions": request.state.n_txn,
            }
        )
    )
    return response


# --------------------------------------------------------------------------- routes
v1 = APIRouter(prefix="/v1")


@v1.get("/health")
def health():
    return {"status": "ok"}


@v1.get("/ready")
def ready():
    try:
        fm = model()
        database()
    except Exception as e:  # noqa: BLE001 - readiness reports any failure
        raise HTTPException(status_code=503, detail=f"Not ready: {type(e).__name__}") from e
    return {"status": "ready", "model_version": fm.version}


@v1.get("/model", dependencies=[Depends(authenticate)])
def model_info():
    fm = model()
    m = load_metrics()
    return {
        "model_version": fm.version,
        "model": fm.model_name,
        "threshold": fm.threshold,
        "bands": fm.band_bounds(),
        "features": fm.features,
        "data_source": fm.data_source,
        "test_metrics": {
            "pr_auc": m["test"]["pr_auc"],
            "pr_auc_ci95": m["test"]["pr_auc_ci95"],
            "recall": m["test"]["at_threshold"]["recall"],
            "precision": m["test"]["at_threshold"]["precision"],
        },
    }


@v1.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest, request: Request, client: str = Depends(authenticate)):
    max_batch = settings().api.max_batch
    if len(req.transactions) > max_batch:
        raise HTTPException(status_code=413, detail=f"At most {max_batch} transactions per request")
    request.state.n_txn = len(req.transactions)
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
    e_cfg = settings().explain
    results = []
    for tn, e in zip(
        df["trans_num"], explain_frame(fm, feat, None, e_cfg.top_raising, e_cfg.top_lowering), strict=True
    ):
        results.append(
            ScoreOut(
                trans_num=tn,
                probability=e.probability,
                band=e.band,
                flagged=e.probability >= fm.threshold,
                action=e.action,
                summary=e.summary,
                reasons=[
                    ReasonOut(text=r.text, direction=r.direction, contribution=r.contribution)
                    for r in e.raising + e.lowering
                ],
            )
        )
    return ScoreResponse(
        model_version=fm.version, threshold=fm.threshold, request_id=request.state.request_id, results=results
    )


app.include_router(v1)


# Unversioned aliases kept for clients of the first release (same handlers).
@app.get("/health", include_in_schema=False)
def health_legacy():
    return health()


@app.post("/score", response_model=list[ScoreOut], include_in_schema=False)
def score_legacy(req: ScoreRequest, request: Request, client: str = Depends(authenticate)):
    return score(req, request, client).results
