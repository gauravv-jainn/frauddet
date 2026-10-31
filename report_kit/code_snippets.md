# Code snippets (verbatim)

Every excerpt below is cut from the file by line range by `report_kit/scripts/make_snippets.py` (no retyping). Line numbers refer to the repository state at commit `f73d82a` plus the uncommitted report_kit/ files. Each excerpt is at most 25 lines.

## S1. Rows are ordered so a row never sees a later row

`fraudlens/fraudkit/features.py` lines 147-165. Leakage prevention: rows are sorted by (card, time, original position); every card feature below is computed on earlier rows only.

```python
def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute all per-row features (except the merchant target encoding).

    Returns a frame aligned with ``raw.index`` containing FEATURES (minus merchant_te),
    CONTEXT columns, plus the original raw columns.
    """
    df = raw.copy()
    df["_pos"] = np.arange(len(df))
    df = df.sort_values(["cc_num", "trans_date_trans_time", "_pos"], kind="mergesort")

    t = df["trans_date_trans_time"]
    amt = df["amt"].to_numpy(dtype="float64")
    df["log_amt"] = np.log1p(amt)
    df["hour"] = t.dt.hour.astype("int16")
    df["day_of_week"] = t.dt.dayofweek.astype("int16")
    df["is_night"] = df["hour"].isin(NIGHT_HOURS).astype("int8")
    df["dist_home_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    df["age"] = ((t - df["dob"]).dt.days / 365.25).astype("float64")
    df["log_city_pop"] = np.log1p(df["city_pop"].astype("float64"))
```

## S2. Rolling window count and sum over EARLIER rows only

`fraudlens/fraudkit/features.py` lines 92-99. `count = idx - lo` excludes the current row (`lo..i-1`); the same helper builds the 1 h / 24 h / 7 d / 30 d velocity features.

```python
def _window_stats(times_ns: np.ndarray, amts: np.ndarray, window_ns: int):
    """For each row i (within one card, time-sorted): count and amount sum of rows j < i with t_j >= t_i - window."""
    lo = np.searchsorted(times_ns, times_ns - window_ns, side="left")
    idx = np.arange(len(times_ns))
    count = idx - lo
    csum = np.concatenate([[0.0], np.cumsum(amts)])
    amt_sum = csum[idx] - csum[lo]
    return count, amt_sum
```

## S3. Merchant / category risk with a 7-day label delay

`fraudlens/fraudkit/features.py` lines 128-135. Fraud labels younger than 7 days are ignored (chargebacks arrive late in production); the rate is smoothed towards a 0.5% prior.

```python
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        ts = t_sorted[a:b]
        lo = np.searchsorted(ts, ts - (RISK_WINDOW_DAYS + LABEL_DELAY_DAYS) * day_ns, side="left")
        hi = np.searchsorted(ts, ts - LABEL_DELAY_DAYS * day_ns, side="left")
        ck = np.r_[0.0, np.cumsum(known[a:b])]
        cf = np.r_[0.0, np.cumsum(fraud[a:b])]
        n_lab, n_fraud = ck[hi] - ck[lo], cf[hi] - cf[lo]
        risk[a:b] = (n_fraud + RISK_PRIOR * RISK_SMOOTHING) / (n_lab + RISK_SMOOTHING)
```

## S4. Out-of-fold merchant target encoding

`fraudlens/fraudkit/features.py` lines 311-320. A fit row never sees its own label through the merchant feature: each fold is encoded with a mapping fitted on the other folds.

```python
    def fit_transform_oof(self, merchant: pd.Series, y: pd.Series) -> np.ndarray:
        self.fit(merchant, y)
        rng = np.random.default_rng(self.seed)
        folds = rng.integers(0, self.n_folds, size=len(merchant))
        out = np.empty(len(merchant), dtype="float64")
        for k in range(self.n_folds):
            tr, te = folds != k, folds == k
            mp, prior = self._fit_map(merchant[tr], y[tr])
            out[te] = merchant[te].map(mp).astype("float64").fillna(prior).to_numpy()
        return out
```

## S5. Time-based fit / validation split and encoder fitting

`fraudlens/fraudkit/train.py` lines 349-358. The validation period is the LAST 15% of the training file by time; the encoder is fitted on the fit rows only.

```python
        with R.stage("split", 1) as st:
            tr_all = feat[feat["_split"] == "train"].sort_values("trans_date_trans_time", kind="mergesort")
            test = feat[feat["_split"] == "test"]
            cut_i = int(len(tr_all) * (1 - float(cfg["data"]["validation_fraction"])))
            cutoff = tr_all["trans_date_trans_time"].iloc[cut_i]
            fit = tr_all[tr_all["trans_date_trans_time"] < cutoff].copy()
            val = tr_all[tr_all["trans_date_trans_time"] >= cutoff].copy()
            test = test.copy()
            enc = MerchantEncoder(smoothing=float(cfg["features"]["te_smoothing"]), seed=seed)
            fit["merchant_te"] = enc.fit_transform_oof(fit["merchant"], fit["is_fraud"])
```

## S6. Optuna objective (3-fold CV PR-AUC)

`fraudlens/fraudkit/train.py` lines 460-477. The 10 tuned LightGBM hyper-parameters and their search ranges; the objective is mean PR-AUC over the folds. Folds are stratified and shuffled (not time-ordered).

```python
        def objective(trial: optuna.Trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 1200, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0, log=True),
            }
            scores = []
            for tr, te in folds:
                m = LGBMClassifier(**params, **lgbm_base(seed)).fit(X_fit.iloc[tr], y_fit[tr])
                scores.append(average_precision_score(y_fit[te], m.predict_proba(X_fit.iloc[te])[:, 1]))
            return float(np.mean(scores))
```

## S7. Calibrator: Platt / beta / isotonic

`fraudlens/fraudkit/score.py` lines 61-85. Three calibration methods behind one class; outputs are clipped to [1e-12, 1 - 1e-12] so a probability is never exactly 0 or 1.

```python
    @staticmethod
    def _design(p_raw: np.ndarray, method: str) -> np.ndarray:
        p = np.clip(np.asarray(p_raw, dtype="float64"), 1e-15, 1 - 1e-15)
        if method == "platt":
            return np.log(p / (1 - p)).reshape(-1, 1)
        return np.column_stack([np.log(p), -np.log1p(-p)])

    def fit(self, p_raw: np.ndarray, y: np.ndarray):
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_raw, y)
        else:
            from sklearn.linear_model import LogisticRegression

            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(self._design(p_raw, self.method), y)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        p_raw = np.asarray(p_raw, dtype="float64")
        if self.method == "isotonic":
            out = self.model.predict(p_raw)
        else:
            out = self.model.predict_proba(self._design(p_raw, self.method))[:, 1]
        return np.clip(out, P_MIN, P_MAX)
```

## S8. 0-999 risk score on a log-odds scale

`fraudlens/fraudkit/score.py` lines 29-39. p = 1e-6 maps to 0 and p = 0.999 to 999; `score_points_to_double_odds` gives the points that double the odds.

```python
SCORE_P_LOW, SCORE_P_HIGH = 1e-6, 0.999


def risk_score(p) -> np.ndarray:
    lo, hi = _logit(SCORE_P_LOW), _logit(SCORE_P_HIGH)
    z = _logit(np.atleast_1d(np.asarray(p, dtype="float64")))
    return np.clip(np.round((z - lo) / (hi - lo) * 999), 0, 999).astype(int)


def score_points_to_double_odds() -> float:
    return float(999 * np.log(2) / (_logit(SCORE_P_HIGH) - _logit(SCORE_P_LOW)))
```

## S9. Decision policy: flags and the four actions

`fraudlens/fraudkit/decision.py` lines 33-52. Threshold mode vs expected-loss mode (p x amount >= k x review cost) and the approve / step_up / review / block logic.

```python
def policy_flags(p, amt, policy: dict) -> np.ndarray:
    """Vectorised: True where the transaction is reviewed or blocked by the model-side policy."""
    p = np.asarray(p, dtype="float64")
    amt = np.asarray(amt, dtype="float64")
    critical = p >= policy["critical"]
    if policy["mode"] == "expected_loss":
        return critical | (p * amt >= policy["k"] * policy["review_cost"])
    return critical | (p >= policy["threshold"])


def decide(p: float, amt: float, band: str, rule_band: str, policy: dict) -> tuple[str, float]:
    """Action for one transaction and its expected loss (p x amount)."""
    el = float(p) * float(amt)
    if band == "Critical" or rule_band == "Critical":
        return "block", el
    if bool(policy_flags([p], [amt], policy)[0]) or rule_band == "High":
        return "review", el
    if band != "Low" or rule_band == "Medium":
        return "step_up", el
    return "approve", el
```

## S10. Policy rules (guardrail on top of the model)

`fraudlens/fraudkit/rules.py` lines 87-111. Vectorised rules (large amount, spend spike, far from home, impossible travel, burst, out-of-range, watchlists); they can only raise a band.

```python
def rule_hits(feat: pd.DataFrame, cfg: dict, ranges: dict | None = None, watch: dict | None = None) -> pd.DataFrame:
    """Vectorised: one boolean column per rule, aligned with ``feat``."""
    out = pd.DataFrame(index=feat.index)
    if not cfg.get("enabled", True):
        return out
    amt = feat["amt"].astype(float)
    out["large_amount"] = amt >= cfg["review_amount"]
    ratio = feat["amt_to_card_mean"].astype(float)
    out["spend_spike"] = (ratio >= cfg["spend_spike_ratio"]) & (amt >= cfg["spend_spike_min_amount"])
    out["far_from_home"] = feat["dist_home_km"].astype(float) >= cfg["far_from_home_km"]
    # The model's own impossible-travel flag is noisy in this data (merchants 50-300 km apart within minutes),
    # so the policy rule only fires for long jumps the training data never contains.
    jump = feat["prev_city_km"].astype(float) if "prev_city_km" in feat else pd.Series(0.0, index=feat.index)
    out["impossible_travel"] = (feat["impossible_travel"].astype(int) == 1) & (jump >= cfg["far_from_home_km"])
    out["burst"] = feat["n_txn_1h"].astype(float) >= cfg["burst_txn_1h"]
    if cfg.get("out_of_range", True) and ranges:
        oor = pd.Series(False, index=feat.index)
        for f in RANGE_FEATURES:
            if f in ranges and f in feat:
                v = feat[f].astype(float)
                oor |= (v > ranges[f]["max"]) | (v < ranges[f]["min"])
        out["out_of_range"] = oor
    if watch:
        out["card_blocked"], out["merchant_watchlist"] = _watch_hits(feat, watch)
    return out
```

## S11. From SHAP values to a reason-coded explanation

`fraudlens/fraudkit/explain.py` lines 347-364. One call: features -> raw score -> calibrated p -> band -> risk score -> SHAP contributions -> reasons -> policy rule hits.

```python
    X = fm.prepare(feat)
    p_raw = fm.predict_raw(X)
    p = fm.calibrator.transform(p_raw)
    bands = fm.band(p, threshold)
    scores = fm.risk_score(p)
    sv, base = shap_matrix(fm, X)
    out = []
    rows = feat.assign(merchant_te=X["merchant_te"].to_numpy())
    for i in range(len(X)):
        contrib = dict(zip(FEATURES, sv[i].tolist(), strict=True))
        row = rows.iloc[i]
        up, down = build_reasons(contrib, row, fm, top_raising, top_lowering)
        model_band = str(bands[i])
        hits = describe_rules(row, policy["rules"], policy.get("ranges"), policy.get("watch")) if policy else []
        final = model_band
        for h in hits:
            final = combine_band(final, h.band)
        summary = summarize(model_band, float(p[i]), up, down)
```

## S12. Population Stability Index and its status bands

`fraudlens/fraudkit/monitoring.py` lines 39-50. PSI = sum((actual - expected) * ln(actual / expected)); < 0.10 stable, 0.10-0.25 moderate, >= 0.25 significant.

```python
def psi(expected: list[float], actual: list[float]) -> float:
    e = np.clip(np.asarray(expected, dtype="float64"), _EPS, None)
    a = np.clip(np.asarray(actual, dtype="float64"), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    if value >= PSI_SIGNIFICANT:
        return "significant"
    if value >= PSI_MODERATE:
        return "moderate"
    return "stable"
```

## S13. Password hashing (PBKDF2-HMAC-SHA256, 600,000 iterations)

`fraudlens/fraudkit/auth.py` lines 60-72. Salted PBKDF2 with a constant-time comparison; the iteration count constant is `PBKDF2_ITERATIONS` in the same file.

```python
def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_b64, hash_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), base64.b64decode(salt_b64), int(iters))
        return hmac.compare_digest(dk, base64.b64decode(hash_b64))
```

## S14. POST /v1/score: validate, then featurize with the card's history

`fraudlens/api/main.py` lines 270-290. Batch-size limit (413), schema validation (422), then features are built from the request rows plus earlier rows of the same cards.

```python
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
    m = load_metrics()
    pol = {
        "rules": resolve_rules(settings().rules.model_dump(), m),
```

## S15. Versioned SQLite migrations

`fraudlens/fraudkit/db.py` lines 123-135. `PRAGMA user_version` tracks the schema; an old database is upgraded in place, each migration inside one transaction.

```python
def migrate(path: str | Path) -> int:
    """Apply pending migrations; returns the schema version."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0 and conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cases'").fetchone():
            version = 1  # database created before migrations existed
        for i in range(version, len(MIGRATIONS)):
            conn.executescript("BEGIN;" + MIGRATIONS[i] + f"PRAGMA user_version={i + 1};COMMIT;")
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
```
