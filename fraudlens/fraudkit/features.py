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
]
FEATURES = NUMERIC + CATEGORICAL

# Context columns carried alongside features (used by explanations and the UI, not by the model).
# card_txn_count is context only: a lifetime counter only ever grows, so it drifts out of the training range
# (PSI > 3 every month on the test period). n_txn_30d carries the "established card" signal and is stationary.
CONTEXT = ["prev_city_km", "minutes_since_prev", "card_mean_amt", "cat_mean_amt", "card_txn_count"]


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
    df["n_txn_1h"] = n1
    df["n_txn_24h"] = n24
    df["n_txn_30d"] = n30d
    df["amt_sum_24h"] = s24

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
