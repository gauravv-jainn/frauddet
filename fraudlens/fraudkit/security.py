"""Small security helpers shared by the app and the API (no Streamlit imports)."""

from __future__ import annotations

import html
import os
import tempfile
from pathlib import Path

import pandas as pd

# Cells starting with these characters are executed as formulas by Excel / Sheets (CSV injection).
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def esc(value) -> str:
    """HTML-escape any value before it is placed inside markup rendered with unsafe_allow_html."""
    return html.escape("" if value is None else str(value), quote=True)


def sanitize_csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralise spreadsheet formulas in text columns before a CSV download (OWASP CSV injection)."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].map(lambda v: "'" + v if isinstance(v, str) and v.startswith(_FORMULA_PREFIXES) else v)
    return out


def mask_card(cc_num) -> str:
    """Only the last four digits of a card number are ever stored or shown outside the raw data."""
    s = str(cc_num)
    return "****" + s[-4:] if len(s) >= 4 else "****"


def atomic_write_bytes(path: Path, data: bytes):
    """Write a file so readers never see a partial version (write temp file, then os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str):
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_path(path: Path):
    """Context-free helper: returns a temp path next to ``path``; call ``os.replace(tmp, path)`` when done."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    return Path(tmp)
