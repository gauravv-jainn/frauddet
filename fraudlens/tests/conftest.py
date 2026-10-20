import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_toy(n_cards: int = 6, n_per_card: int = 40, seed: int = 0) -> pd.DataFrame:
    """Small synthetic frame with the Kaggle schema (no download needed for tests)."""
    rng = np.random.default_rng(seed)
    rows = []
    cats = ["grocery_pos", "shopping_net", "gas_transport", "misc_net", "travel"]
    for c in range(n_cards):
        lat, lon = rng.uniform(30, 45), rng.uniform(-110, -75)
        t = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 20 * 24 * 3600, n_per_card)), unit="s")
        # force some exact timestamp ties and very close transactions
        t = t.to_numpy().copy()
        t[5] = t[4]
        for i in range(n_per_card):
            rows.append({
                "trans_date_trans_time": pd.Timestamp(t[i]),
                "cc_num": 4000_0000_0000_0000 + c,
                "merchant": f"m{rng.integers(0, 15)}",
                "category": cats[rng.integers(0, len(cats))],
                "amt": float(np.round(rng.lognormal(3.5, 1.0), 2)),
                "gender": "F" if c % 2 else "M",
                "city": f"city{c}", "state": "NY",
                "lat": lat, "long": lon, "city_pop": int(rng.integers(500, 2_000_000)),
                "job": "Engineer", "dob": pd.Timestamp("1980-05-05"),
                "merch_lat": lat + rng.normal(0, 3 if i % 7 == 0 else 0.5),
                "merch_long": lon + rng.normal(0, 3 if i % 7 == 0 else 0.5),
                "trans_num": f"t{c}_{i}",
                "is_fraud": int(rng.random() < 0.1),
            })
    df = pd.DataFrame(rows)
    # interleave cards in time order like the real file
    return df.sort_values("trans_date_trans_time", kind="mergesort").reset_index(drop=True)


@pytest.fixture
def toy():
    return make_toy()
