"""Leakage-free feature pipeline.

Every card-behaviour feature for a transaction is computed only from EARLIER
transactions of the same card (strictly earlier in the per-card order). Rows are
ordered by (cc_num, time, original position) so ties are resolved
deterministically and a row never sees a row that comes after it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NIGHT_HOURS = {22, 23, 0, 1, 2, 3}
IMPOSSIBLE_SPEED_KMH = 800.0  # faster than a commercial flight
SMALL_AMOUNT = 5.0  # card-testing probes are typically a few cents to < $5
LABEL_DELAY_DAYS = 7  # fraud labels (chargebacks) are assumed known one week after the transaction
RISK_WINDOW_DAYS = 30
RISK_PRIOR, RISK_SMOOTHING = 0.005, 50.0  # smoothed fraud rate pulls small entities towards a 0.5% prior

# Candidate features from the literature, tested in scripts/feature_ablation.py before being used by the model.
RESEARCH_FEATURES = {
    "spending windows (RFM)": ["n_txn_7d", "avg_amt_1d", "avg_amt_7d", "avg_amt_30d", "amt_to_avg_30d"],
    "amount z-score vs card history": ["amt_log_z"],
    "card-testing probes": ["small_txn_24h"],
    "weekend + circular hour": ["is_weekend", "hour_sin", "hour_cos"],
    "merchant/category fraud risk (7-day label delay)": ["merch_risk_30d", "cat_risk_30d"],
    "merchant velocity": ["merch_txn_1h", "merch_txn_24h"],
}

CATEGORICAL = ["category", "gender"]
NUMERIC = [
    "amt",
    "log_amt",
    "hour",
    "day_of_week",
    "is_night",
    "dist_home_km",
    "age",
    "log_city_pop",
    "merchant_te",
    "n_txn_1h",
    "n_txn_24h",
    "amt_sum_24h",
    "amt_to_card_mean",
    "amt_to_cat_mean",
    "hours_since_prev",
    "geo_speed_kmh",
    "impossible_travel",
    "first_time_category",
    "n_txn_30d",
    # added after the validation ablation (scripts/feature_ablation.py): +0.0017 PR-AUC, 3x the seed noise
    "n_txn_7d",
    "avg_amt_1d",
    "avg_amt_7d",
    "avg_amt_30d",
    "amt_to_avg_30d",
    "amt_log_z",
    "small_txn_24h",
]
FEATURES = NUMERIC + CATEGORICAL

# Context columns carried alongside features (used by explanations and the UI, not by the model).
# card_txn_count is context only: a lifetime counter only ever grows, so it drifts out of the training range
# (PSI > 3 every month on the test period). n_txn_30d carries the "established card" signal and is stationary.
# card_max_amt / amt_to_card_max / cat_share / hour_share were tested as model features: none improved validation
# PR-AUC beyond seed noise (+-0.0005), so they stay context only (explanations, policy rules, card 360 view).
CONTEXT = [
    "prev_city_km",
    "minutes_since_prev",
    "card_mean_amt",
    "cat_mean_amt",
    "card_txn_count",
    "card_max_amt",
    "amt_to_card_max",
    "cat_share",
    "hour_share",
]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype="float64")) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _finite(a) -> np.ndarray:
    a = np.asarray(a, dtype="float64")
    return np.where(np.isfinite(a), a, np.nan)


def _window_stats(times_ns: np.ndarray, amts: np.ndarray, window_ns: int):
    """For each row i (within one card, time-sorted): count and amount sum of rows j < i with t_j >= t_i - window."""
    lo = np.searchsorted(times_ns, times_ns - window_ns, side="left")
    idx = np.arange(len(times_ns))
    count = idx - lo
    csum = np.concatenate([[0.0], np.cumsum(amts)])
    amt_sum = csum[idx] - csum[lo]
    return count, amt_sum


def _entity_features(
    df: pd.DataFrame, times_ns: np.ndarray, amt: np.ndarray, col: str, prefix: str, velocity: bool = True
):
    """Per-entity (merchant / category) features from earlier rows of the same entity.

    * ``{prefix}_txn_1h`` / ``_24h``: transactions at the entity in the last hour / day (all cards).
    * ``{prefix}_risk_30d``: smoothed fraud rate of the entity over [t - 37 d, t - 7 d). Labels younger than
      LABEL_DELAY are ignored because in production chargebacks arrive late (Fraud Detection Handbook).
      Rows without a label (e.g. newly scored transactions) count neither as fraud nor as volume.
    """
    codes = pd.factorize(df[col].astype(str))[0]
    order = np.lexsort((df["_pos"].to_numpy(), times_ns, codes))  # ties: original arrival order
    c_sorted, t_sorted = codes[order], times_ns[order]
    y = (
        pd.to_numeric(df["is_fraud"], errors="coerce").to_numpy(dtype="float64")
        if "is_fraud" in df
        else np.full(len(df), np.nan)
    )
    y_sorted = y[order]
    known = (~np.isnan(y_sorted)).astype("float64")
    fraud = np.nan_to_num(y_sorted)
    bounds = np.flatnonzero(np.r_[True, c_sorted[1:] != c_sorted[:-1], True])
    hour_ns, day_ns = 3_600 * 1_000_000_000, 24 * 3_600 * 1_000_000_000
    risk = np.empty(len(df))
    v1 = np.zeros(len(df), dtype="int32")
    v24 = np.zeros(len(df), dtype="int32")
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        ts = t_sorted[a:b]
        lo = np.searchsorted(ts, ts - (RISK_WINDOW_DAYS + LABEL_DELAY_DAYS) * day_ns, side="left")
        hi = np.searchsorted(ts, ts - LABEL_DELAY_DAYS * day_ns, side="left")
        ck = np.r_[0.0, np.cumsum(known[a:b])]
        cf = np.r_[0.0, np.cumsum(fraud[a:b])]
        n_lab, n_fraud = ck[hi] - ck[lo], cf[hi] - cf[lo]
        risk[a:b] = (n_fraud + RISK_PRIOR * RISK_SMOOTHING) / (n_lab + RISK_SMOOTHING)
        if velocity:
            v1[a:b], _ = _window_stats(ts, amt[order][a:b], hour_ns)
            v24[a:b], _ = _window_stats(ts, amt[order][a:b], day_ns)
    inv = np.empty(len(df), dtype=np.int64)
    inv[order] = np.arange(len(df))
    df[f"{prefix}_risk_30d"] = risk[inv]
    if velocity:
        df[f"{prefix}_txn_1h"] = v1[inv]
        df[f"{prefix}_txn_24h"] = v24[inv]


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

    g = df.groupby("cc_num", sort=False)
    # history counts / expanding mean of PREVIOUS amounts
    df["card_txn_count"] = g.cumcount().astype("int32")
    prev_sum = g["amt"].cumsum() - df["amt"]
    with np.errstate(divide="ignore", invalid="ignore"):
        card_mean = np.where(df["card_txn_count"] > 0, prev_sum / df["card_txn_count"], np.nan)
        df["amt_to_card_mean"] = _finite(amt / card_mean)  # an all-zero history would give inf
    df["card_mean_amt"] = card_mean

    gc = df.groupby(["cc_num", "category"], sort=False, observed=True)
    cat_count = gc.cumcount()
    cat_prev_sum = gc["amt"].cumsum() - df["amt"]
    with np.errstate(divide="ignore", invalid="ignore"):
        cat_mean = np.where(cat_count > 0, cat_prev_sum / cat_count, np.nan)
        df["amt_to_cat_mean"] = _finite(amt / cat_mean)
    df["cat_mean_amt"] = cat_mean
    # first time this card buys in this category (only meaningful once the card has some history)
    df["first_time_category"] = (cat_count == 0).astype("int8")

    # time since previous transaction and geo-speed from the previous merchant location
    prev_t = g["trans_date_trans_time"].shift(1)
    dt_h = (t - prev_t).dt.total_seconds() / 3600.0
    df["hours_since_prev"] = dt_h
    df["minutes_since_prev"] = dt_h * 60.0
    prev_lat = g["merch_lat"].shift(1)
    prev_lon = g["merch_long"].shift(1)
    jump_km = np.asarray(haversine_km(prev_lat, prev_lon, df["merch_lat"], df["merch_long"]))
    df["prev_city_km"] = jump_km
    speed = jump_km / np.maximum(dt_h.to_numpy(), 1.0 / 60.0)  # at least one minute apart
    df["geo_speed_kmh"] = speed
    with np.errstate(invalid="ignore"):
        df["impossible_travel"] = ((speed > IMPOSSIBLE_SPEED_KMH) & (jump_km > 50)).astype("int8")  # NaN compares False

    # velocity windows
    times_ns = t.to_numpy().astype("datetime64[ns]").astype("int64")
    cards = df["cc_num"].to_numpy()
    n1 = np.zeros(len(df), dtype="int32")
    n24 = np.zeros(len(df), dtype="int32")
    n30d = np.zeros(len(df), dtype="int32")
    s24 = np.zeros(len(df), dtype="float64")
    bounds = np.flatnonzero(np.r_[True, cards[1:] != cards[:-1], True])
    hour_ns = 3_600 * 1_000_000_000
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        n1[a:b], _ = _window_stats(times_ns[a:b], amt[a:b], hour_ns)
        n24[a:b], s24[a:b] = _window_stats(times_ns[a:b], amt[a:b], 24 * hour_ns)
        n30d[a:b], _ = _window_stats(times_ns[a:b], amt[a:b], 30 * 24 * hour_ns)
    # card profile: previous maximum spend, how usual this hour is for the card (share of earlier transactions
    # within +-2 hours, circular) -- both from strictly earlier rows of the card
    hours = df["hour"].to_numpy()
    prev_max = np.full(len(df), np.nan)
    hour_share = np.full(len(df), np.nan)
    offsets = np.arange(-2, 3)
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        seg = amt[a:b]
        run_max = np.maximum.accumulate(seg)
        prev_max[a + 1 : b] = run_max[:-1]
        onehot = np.zeros((b - a, 24), dtype="int32")
        onehot[np.arange(b - a), hours[a:b]] = 1
        before = np.cumsum(onehot, axis=0) - onehot  # counts of earlier transactions per hour
        cols = (hours[a:b, None] + offsets[None, :]) % 24
        near = np.take_along_axis(before, cols, axis=1).sum(axis=1)
        n_prev = np.arange(b - a)
        with np.errstate(divide="ignore", invalid="ignore"):
            hour_share[a:b] = np.where(n_prev > 0, near / n_prev, np.nan)
    df["card_max_amt"] = prev_max
    with np.errstate(divide="ignore", invalid="ignore"):
        df["amt_to_card_max"] = _finite(amt / prev_max)
        df["cat_share"] = np.where(df["card_txn_count"] > 0, cat_count / df["card_txn_count"], np.nan)
    df["hour_share"] = hour_share

    df["n_txn_1h"] = n1
    df["n_txn_24h"] = n24
    df["n_txn_30d"] = n30d
    df["amt_sum_24h"] = s24

    # ---- research-based features (see PROGRESS M11 for sources); all from strictly earlier rows ----
    # RFM-style spending windows (Fraud Detection Handbook): count and average amount over 1 / 7 / 30 days
    day_ns = 24 * hour_ns
    n7 = np.zeros(len(df), dtype="int32")
    s1, s7, s30, small24 = (np.zeros(len(df)) for _ in range(4))
    small = (amt < SMALL_AMOUNT).astype("float64")
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        _, s1[a:b] = _window_stats(times_ns[a:b], amt[a:b], day_ns)
        n7[a:b], s7[a:b] = _window_stats(times_ns[a:b], amt[a:b], 7 * day_ns)
        _, s30[a:b] = _window_stats(times_ns[a:b], amt[a:b], 30 * day_ns)
        _, small24[a:b] = _window_stats(times_ns[a:b], small[a:b], day_ns)  # card-testing probes
    with np.errstate(divide="ignore", invalid="ignore"):
        avg30 = np.where(n30d > 0, s30 / n30d, np.nan)
        df["n_txn_7d"] = n7
        df["avg_amt_1d"] = np.where(n24 > 0, s1 / n24, np.nan)
        df["avg_amt_7d"] = np.where(n7 > 0, s7 / n7, np.nan)
        df["avg_amt_30d"] = avg30
        df["amt_to_avg_30d"] = _finite(amt / avg30)
        # z-score of log(1 + amount) against the card's own history (expanding mean / std of earlier rows)
        la = np.log1p(amt)
        n_prev = df["card_txn_count"].to_numpy().astype("float64")
        csum = df.assign(_la=la).groupby("cc_num", sort=False)["_la"].cumsum().to_numpy() - la
        csq = df.assign(_la2=la * la).groupby("cc_num", sort=False)["_la2"].cumsum().to_numpy() - la * la
        mean = csum / n_prev
        std = np.sqrt(np.maximum(csq / n_prev - mean * mean, 0.0))
        df["amt_log_z"] = np.where((n_prev >= 2) & (std > 1e-6), (la - mean) / std, np.nan)
    df["small_txn_24h"] = small24.astype("int32")
    # calendar (weekend) and time as a circle
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    # merchant velocity (all cards) and merchant / category fraud risk from labels at least LABEL_DELAY old
    _entity_features(df, times_ns, amt, "merchant", "merch")
    _entity_features(df, times_ns, amt, "category", "cat", velocity=False)

    df["category"] = df["category"].astype("string")
    df["gender"] = df["gender"].astype("string")
    df = df.sort_values("_pos", kind="mergesort").drop(columns="_pos")
    df.index = raw.index
    return df


class MerchantEncoder:
    """Smoothed target encoding of merchant, fitted on the fit split only.

    ``fit_transform_oof`` gives out-of-fold encodings for the fit rows themselves so
    the model never sees a row's own label through this feature.
    """

    def __init__(self, smoothing: float = 50.0, n_folds: int = 5, seed: int = 42):
        self.smoothing = smoothing
        self.n_folds = n_folds
        self.seed = seed
        self.mapping_: dict[str, float] = {}
        self.prior_: float = 0.0

    def _fit_map(self, merchant: pd.Series, y: pd.Series):
        prior = float(y.mean())
        stats = pd.DataFrame({"m": merchant.to_numpy(), "y": y.to_numpy()}).groupby("m")["y"].agg(["sum", "count"])
        enc = (stats["sum"] + self.smoothing * prior) / (stats["count"] + self.smoothing)
        return enc.to_dict(), prior

    def fit(self, merchant: pd.Series, y: pd.Series):
        self.mapping_, self.prior_ = self._fit_map(merchant, y)
        return self

    def transform(self, merchant: pd.Series) -> np.ndarray:
        return merchant.map(self.mapping_).astype("float64").fillna(self.prior_).to_numpy()

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


def model_frame(feat: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Select FEATURES in a fixed order with fixed categorical levels (for LightGBM/SHAP)."""
    X = feat[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c].astype("object"), categories=categories[c])
    for c in NUMERIC:
        X[c] = X[c].astype("float64")
    return X


def featurize_with_history(new_rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Features for ``new_rows`` given earlier transactions in ``history``.

    Only history rows of the same cards that happened strictly before each new row can
    influence it (guaranteed by build_features' ordering). Returns rows aligned to new_rows.
    """
    cards = new_rows["cc_num"].unique()
    hist = history[history["cc_num"].isin(cards)]
    hist = hist[hist["trans_date_trans_time"] < new_rows["trans_date_trans_time"].max()]
    combo = pd.concat([hist, new_rows], ignore_index=True)
    combo["_new"] = np.r_[np.zeros(len(hist), bool), np.ones(len(new_rows), bool)]
    feats = build_features(combo)
    out = feats[feats["_new"]].drop(columns="_new")
    out.index = new_rows.index
    return out
