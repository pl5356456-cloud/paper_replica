"""MSASI 2.0 — Daily signal writer.

Generates the latest MSASI snapshot and writes:
  1. Cumulative daily history (full timeseries)  →  cn/equities/sentiment/msasi/msasi_daily.parquet
  2. Cumulative signal log                        →  cn/equities/sentiment/msasi/msasi_signals.parquet
  3. Latest component diagnostics                 →  cn/equities/sentiment/msasi/latest_*.parquet

Usage (from cron / CLI)::

    from msasi.signal import generate_daily_signal
    result = generate_daily_signal("2026-04-13")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .paths import data_root, data_path

from .config import OVERBOUGHT_THRESHOLD, OVERSOLD_THRESHOLD
from .engine import MSASIEngine, MSASIResult

logger = logging.getLogger(__name__)

__all__ = ["generate_daily_signal", "MSASIDailySnapshot"]


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

def _msasi_output_dir() -> Path:
    """Output directory for MSASI timeseries and diagnostics."""
    return data_path("cn", "equities", "sentiment", "msasi")


# ---------------------------------------------------------------------------
# Zone classification
# ---------------------------------------------------------------------------

def classify_zone(msasi_1mma: float) -> str:
    """Classify MSASI 1MMA into sentiment zone."""
    if np.isnan(msasi_1mma):
        return "UNKNOWN"
    if msasi_1mma <= OVERSOLD_THRESHOLD:
        return "OVERSOLD"
    elif msasi_1mma >= OVERBOUGHT_THRESHOLD:
        return "OVERBOUGHT"
    return "NEUTRAL"


def zone_distance(msasi_1mma: float) -> float:
    """Distance from nearest extreme threshold (negative = inside zone)."""
    if np.isnan(msasi_1mma):
        return np.nan
    dist_oversold = msasi_1mma - OVERSOLD_THRESHOLD
    dist_overbought = OVERBOUGHT_THRESHOLD - msasi_1mma
    return min(dist_oversold, dist_overbought)


# ---------------------------------------------------------------------------
# Snapshot container
# ---------------------------------------------------------------------------

class MSASIDailySnapshot:
    """Container for a single day's MSASI production output."""

    def __init__(
        self,
        date: pd.Timestamp,
        msasi_raw: float,
        msasi_1mma: float,
        zone: str,
        zone_distance: float,
        active_signal: Optional[str],
        weights: pd.Series,
        component_values: dict[str, float],
        full_result: MSASIResult,
    ):
        self.date = date
        self.msasi_raw = msasi_raw
        self.msasi_1mma = msasi_1mma
        self.zone = zone
        self.zone_distance = zone_distance
        self.active_signal = active_signal
        self.weights = weights
        self.component_values = component_values
        self.full_result = full_result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_daily_signal(
    asof_date: Optional[str] = None,
    save: bool = True,
) -> MSASIDailySnapshot:
    """Run MSASI engine and produce daily snapshot.

    Parameters
    ----------
    asof_date : str, optional
        Date string (YYYY-MM-DD). If None, uses latest available.
    save : bool
        If True, write outputs to disk.

    Returns
    -------
    MSASIDailySnapshot
    """
    from .data.readers import load_all_indicators, load_allshare_prices

    logger.info("=" * 60)
    logger.info("MSASI 2.0 — Daily Signal Generation")
    logger.info("=" * 60)

    # Load data
    raw = load_all_indicators(start_date="2013-01-01")
    allshare = load_allshare_prices()

    if raw.empty:
        raise RuntimeError("No indicator data loaded — cannot generate signal.")

    # Run engine (full history for expanding weights)
    engine = MSASIEngine()
    result = engine.run(raw, allshare, use_expanding_weights=True)

    # Extract latest day
    daily = result.msasi_daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    if asof_date is not None:
        target = pd.Timestamp(asof_date)
        # Find closest trading day on or before target
        candidates = daily[daily["date"] <= target]
        if candidates.empty:
            raise ValueError(f"No MSASI data on or before {asof_date}")
        latest = candidates.iloc[-1]
    else:
        latest = daily.iloc[-1]

    latest_date = latest["date"]
    msasi_raw = float(latest["msasi_raw"])
    msasi_1mma = float(latest["msasi_1mma"])
    zone = classify_zone(msasi_1mma)
    zone_dist = zone_distance(msasi_1mma)

    # Check if there's an active signal on this date
    active_signal = None
    if not result.signals.empty:
        sig_match = result.signals[
            result.signals["date"] == latest_date
        ]
        if not sig_match.empty:
            active_signal = sig_match.iloc[0]["signal_type"]

    # Extract component values for latest date
    raw_comp = result.component_raw
    raw_comp["date"] = pd.to_datetime(raw_comp["date"])
    norm_latest = raw_comp[raw_comp["date"] == latest_date]
    component_values = {}
    if not norm_latest.empty:
        row = norm_latest.iloc[0]
        for col in norm_latest.columns:
            if col != "date" and not np.isnan(row[col]) if isinstance(row[col], float) else True:
                component_values[col] = float(row[col]) if col != "date" else None
        component_values = {k: v for k, v in component_values.items() if v is not None}

    snapshot = MSASIDailySnapshot(
        date=latest_date,
        msasi_raw=msasi_raw,
        msasi_1mma=msasi_1mma,
        zone=zone,
        zone_distance=zone_dist,
        active_signal=active_signal,
        weights=result.weights,
        component_values=component_values,
        full_result=result,
    )

    logger.info(
        "MSASI snapshot: date=%s raw=%.1f%% 1MMA=%.1f%% zone=%s signal=%s",
        latest_date.date(), msasi_raw, msasi_1mma, zone,
        active_signal or "—",
    )

    if save:
        _save_outputs(snapshot, result)
        try:
            from .shadow import run_shadow
            run_shadow(result.component_normalized, allshare, out_dir=_msasi_output_dir())
        except Exception:  # the shadow must never block the production score
            logger.exception("MSASI shadow (PCA) failed")

    return snapshot


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def _save_outputs(snapshot: MSASIDailySnapshot, result: MSASIResult) -> None:
    """Write all output files."""
    date_str = snapshot.date.strftime("%Y-%m-%d")

    # 1. Full timeseries files
    _save_timeseries(result)

    # 2. Component + weight snapshot
    _save_component_snapshot(snapshot, date_str)


def _save_timeseries(result: MSASIResult) -> None:
    """Save full MSASI timeseries and cumulative signals."""
    out_dir = _msasi_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Daily timeseries
    daily_path = out_dir / "msasi_daily.parquet"
    daily = result.msasi_daily.copy()
    daily["zone"] = daily["msasi_1mma"].apply(classify_zone)
    daily.to_parquet(daily_path, engine="pyarrow", index=False)
    logger.info("Daily timeseries: %d rows → %s", len(daily), daily_path)

    # Signals
    signals_path = out_dir / "msasi_signals.parquet"
    result.signals.to_parquet(signals_path, engine="pyarrow", index=False)
    logger.info("Signals: %d records → %s", len(result.signals), signals_path)


def _save_component_snapshot(snapshot: MSASIDailySnapshot, date_str: str) -> None:
    """Save component-level diagnostics for the latest date."""
    out_dir = _msasi_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Weights
    weights_path = out_dir / "latest_weights.parquet"
    wdf = pd.DataFrame({
        "indicator": snapshot.weights.index,
        "weight": snapshot.weights.values,
        "date": snapshot.date,
    })
    wdf.to_parquet(weights_path, engine="pyarrow", index=False)

    # Component normalized values (latest day)
    if snapshot.component_values:
        comp_path = out_dir / "latest_components.parquet"
        cdf = pd.DataFrame([{
            "date": snapshot.date,
            **snapshot.component_values,
        }])
        cdf.to_parquet(comp_path, engine="pyarrow", index=False)

    logger.info("Component diagnostics saved to %s", out_dir)
