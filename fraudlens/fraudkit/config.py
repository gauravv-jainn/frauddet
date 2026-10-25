"""Typed, validated configuration (configs/default.yaml). A bad value fails at start-up with a clear message."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataCfg(_Strict):
    cache_dir: str | None = None
    validation_fraction: float = Field(0.15, gt=0.0, lt=0.5)


class FeaturesCfg(_Strict):
    merchant_encoding: str = "target"
    te_smoothing: float = Field(50.0, ge=0.0)


class ModelCfg(_Strict):
    optuna_trials: int = Field(30, ge=1, le=1000)
    cv_folds: int = Field(3, ge=2, le=10)
    tuning_sample_negatives: float = Field(0.25, gt=0.0, le=1.0)


class CostsCfg(_Strict):
    review_cost: float = Field(5.0, ge=0.0)


class BandsCfg(_Strict):
    medium: float | None = Field(None, gt=0.0, lt=1.0)  # None = the money-optimal threshold
    high: float = Field(0.30, gt=0.0, lt=1.0)
    critical: float = Field(0.70, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self):
        if self.high >= self.critical:
            raise ValueError("bands.high must be below bands.critical")
        if self.medium is not None and self.medium >= self.high:
            raise ValueError("bands.medium must be below bands.high")
        return self


class ExplainCfg(_Strict):
    top_raising: int = Field(4, ge=1, le=10)
    top_lowering: int = Field(2, ge=0, le=10)


Auto = Literal["auto"]


class RulesCfg(_Strict):
    """Policy guardrails on top of the model. Rules can only raise the risk band.

    "auto" = the value derived at training time (artifacts/metrics.json -> rules_auto): amount and distance limits
    from the training data, the spending-spike rule tuned on the validation period.
    """

    enabled: bool = True
    review_amount: float | Auto = "auto"
    spend_spike_ratio: float | Auto = "auto"
    spend_spike_min_amount: float | Auto = "auto"
    far_from_home_km: float | Auto = "auto"
    burst_txn_1h: int = Field(5, ge=2)
    out_of_range: bool = True
    block_confirmed_cards: bool = True  # an analyst-confirmed fraud blocks the card for later transactions

    @model_validator(mode="after")
    def _positive(self):
        for name in ("review_amount", "spend_spike_ratio", "spend_spike_min_amount", "far_from_home_km"):
            v = getattr(self, name)
            if v != "auto" and v <= 0:
                raise ValueError(f"rules.{name} must be positive or 'auto'")
        return self


class DecisionCfg(_Strict):
    """How a calibrated probability becomes an action (see fraudkit/decision.py)."""

    mode: Literal["auto", "expected_loss", "threshold"] = "auto"  # auto = cheaper policy on validation
    sla_hours: float = Field(24.0, gt=0)  # cases open longer than this are overdue


class AppCfg(_Strict):
    db: str = "data/fraudlens.sqlite"
    session_timeout_minutes: int = Field(60, ge=5, le=24 * 60)
    max_failed_logins: int = Field(5, ge=3, le=20)
    lockout_minutes: int = Field(15, ge=1, le=24 * 60)
    max_upload_rows: int = Field(20_000, ge=100, le=1_000_000)


class ApiCfg(_Strict):
    max_batch: int = Field(1000, ge=1, le=10_000)
    rate_limit_per_minute: int = Field(120, ge=1, le=100_000)


class Settings(_Strict):
    seed: int = 42
    data: DataCfg = DataCfg()
    features: FeaturesCfg = FeaturesCfg()
    model: ModelCfg = ModelCfg()
    costs: CostsCfg = CostsCfg()
    bands: BandsCfg = BandsCfg()
    explain: ExplainCfg = ExplainCfg()
    rules: RulesCfg = RulesCfg()
    decision: DecisionCfg = DecisionCfg()
    app: AppCfg = AppCfg()
    api: ApiCfg = ApiCfg()

    def db_path(self) -> Path:
        """SQLite file for users, API keys, cases and audit. FRAUDLENS_DB overrides the config."""
        p = Path(os.environ.get("FRAUDLENS_DB") or self.app.db)
        return p if p.is_absolute() else ROOT / p


def load_config(path: str | os.PathLike | None = None) -> Settings:
    path = Path(path or os.environ.get("FRAUDLENS_CONFIG") or DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Settings.model_validate(raw)
