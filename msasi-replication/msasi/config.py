"""MSASI 2.0 — Configuration.

All hyperparameters, indicator metadata, and strategy thresholds in one place.
Faithfully replicates the MS paper's methodology.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─── Normalization ────────────────────────────────────────────────
NORM_WINDOW = 100          # 100-day rolling min-max window (paper spec)

# ─── Aggregation ──────────────────────────────────────────────────
MMA_WINDOW = 20            # 1-month moving average ≈ 20 trading days

# ─── Signal Thresholds ───────────────────────────────────────────
OVERBOUGHT_THRESHOLD = 75  # MSASI 1MMA ≥ 75 → short signal
OVERSOLD_THRESHOLD = 25    # MSASI 1MMA ≤ 25 → long signal
COOLDOWN_DAYS = 20         # De-dupe: only first trigger within 1 month

# ─── Backtest Strategy ───────────────────────────────────────────
HOLDING_PERIOD = 70        # Trading days per trade (paper spec)

# ─── Weight Estimation ───────────────────────────────────────────
# R² training window for weight estimation.
# Use expanding window: grow from this anchor forward.
WEIGHT_ANCHOR_START = "2014-01-01"
# Minimum training history before weights become reliable
WEIGHT_MIN_HISTORY_DAYS = 504  # ~2 years

# Forward return horizon for R² computation (in trading days)
WEIGHT_FWD_RETURN_DAYS = 20


# ─── Indicator Registry ──────────────────────────────────────────

@dataclass(frozen=True)
class IndicatorSpec:
    """Metadata for a single MSASI component indicator."""
    name: str               # Human-readable name
    key: str                # Column name in indicator DataFrame
    frequency: str          # 'daily', 'monthly', 'weekly'
    direction: int          # +1 = higher → more bullish, -1 = inverted
    available_from: Optional[str] = None  # Earliest data date
    weight_penalty: float = 1.0  # Multiplier on R² weight (< 1 for penalized)


INDICATOR_SPECS: List[IndicatorSpec] = [
    IndicatorSpec(
        name="ChiNext Turnover",
        key="chinext_turnover",
        frequency="daily",
        direction=1,
    ),
    IndicatorSpec(
        name="A-share Turnover",
        key="ashare_turnover",
        frequency="daily",
        direction=1,
    ),
    IndicatorSpec(
        name="Equity Futures Turnover",
        key="futures_turnover",
        frequency="daily",
        direction=1,
    ),
    IndicatorSpec(
        name="Northbound Turnover",
        key="northbound_turnover",
        frequency="daily",
        direction=1,
    ),
    IndicatorSpec(
        name="Margin Outstanding",
        key="margin_balance",
        frequency="daily",
        direction=1,
    ),
    IndicatorSpec(
        name="SSE New Accounts",
        key="sse_new_accounts",
        frequency="monthly",
        direction=1,
        available_from="2021-01-01",
        weight_penalty=0.5,  # Monthly frequency → penalized per paper
    ),
    IndicatorSpec(
        name="RSI-30D (CSI300)",
        key="rsi_30d",
        frequency="daily",
        direction=1,
    ),
    IndicatorSpec(
        name="Limit-Up Count",
        key="limit_up_count",
        frequency="daily",
        direction=1,
    ),
    IndicatorSpec(
        name="CSI300 Futures Backwardation",
        key="futures_basis",
        frequency="daily",
        direction=1,  # Higher basis = more bullish futures premium
    ),
    IndicatorSpec(
        name="CSI300 Call/Put Ratio",
        key="call_put_ratio",
        frequency="daily",
        direction=1,
        available_from="2019-12-01",
        weight_penalty=0.5,  # Short history → penalized per paper
    ),
    IndicatorSpec(
        name="Foreign Passive Flows (1MMA)",
        key="foreign_passive_flow_1mma",
        frequency="daily",
        direction=1,
    ),
]

# Quick lookup
INDICATOR_KEYS: List[str] = [s.key for s in INDICATOR_SPECS]
INDICATOR_MAP: Dict[str, IndicatorSpec] = {s.key: s for s in INDICATOR_SPECS}
N_INDICATORS = len(INDICATOR_SPECS)


# ─── Limit-Up Detection (board-specific thresholds) ──────────────
# The STAR Market (688xxx) and ChiNext (300xxx) have 20% daily limits.
# Main board stocks have 10% limits.
LIMIT_UP_THRESHOLDS = {
    "default": 0.095,      # Main board: 10% limit, detect at 9.5%
    "300": 0.195,          # ChiNext: 20% limit, detect at 19.5%
    "688": 0.195,          # STAR Market: 20% limit, detect at 19.5%
}
