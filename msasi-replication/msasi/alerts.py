"""MSASI 2.0 — Sentiment extreme detection and alerting.

Detects when MSASI 1MMA crosses into oversold/overbought zones and
generates alerts.

Usage (from cron)::

    from msasi.alerts import check_sentiment_extreme
    alert = check_sentiment_extreme(snapshot)
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .paths import data_root, data_path

from .config import OVERBOUGHT_THRESHOLD, OVERSOLD_THRESHOLD

logger = logging.getLogger(__name__)

__all__ = [
    "check_sentiment_extreme",
    "format_sentiment_alert",
    "get_previous_msasi",
    "save_alert_log",
]


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def _msasi_daily_path() -> Path:
    return (
        data_path("cn", "equities", "sentiment", "msasi")
        / "msasi_daily.parquet"
    )


def _alert_log_path() -> Path:
    return (
        data_path("cn", "equities", "sentiment", "msasi")
        / "sentiment_alerts.jsonl"
    )


def get_previous_msasi(as_of_date: date) -> Optional[dict]:
    """Get the most recent MSASI values prior to as_of_date.

    Returns
    -------
    dict or None
        ``{"date": ..., "msasi_raw": ..., "msasi_1mma": ..., "zone": ...}``
    """
    path = _msasi_daily_path()
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    prior = df[df["date"] < pd.Timestamp(as_of_date)]
    if prior.empty:
        return None

    row = prior.sort_values("date").iloc[-1]
    from .signal import classify_zone

    return {
        "date": row["date"],
        "msasi_raw": float(row["msasi_raw"]),
        "msasi_1mma": float(row["msasi_1mma"]),
        "zone": classify_zone(float(row["msasi_1mma"])),
    }


# ---------------------------------------------------------------------------
# Extreme detection
# ---------------------------------------------------------------------------

def check_sentiment_extreme(
    current_date: date,
    current_msasi_1mma: float,
    current_zone: str,
    previous: Optional[dict] = None,
) -> Optional[dict]:
    """Check if a sentiment extreme event occurred.

    An alert fires when:
    - MSASI 1MMA freshly enters OVERSOLD or OVERBOUGHT zone
      (i.e., previous day was NEUTRAL and today is extreme)

    Parameters
    ----------
    current_date : date
        Today's date.
    current_msasi_1mma : float
        Today's MSASI 1MMA value.
    current_zone : str
        Today's zone classification.
    previous : dict, optional
        Previous day's MSASI info. Loaded from disk if not provided.

    Returns
    -------
    dict or None
        Alert info if extreme detected, None otherwise.
    """
    if previous is None:
        previous = get_previous_msasi(current_date)

    if previous is None:
        logger.info("No previous MSASI data — first run, no alert.")
        return None

    prev_zone = previous.get("zone", "NEUTRAL")

    # Only fire on zone entry (not while staying in zone)
    if current_zone == prev_zone:
        logger.info(
            "MSASI zone unchanged: %s (1MMA=%.1f%%)",
            current_zone, current_msasi_1mma,
        )
        return None

    if current_zone == "NEUTRAL":
        # Exiting extreme zone — informational only
        logger.info(
            "MSASI exiting %s zone → NEUTRAL (1MMA=%.1f%%)",
            prev_zone, current_msasi_1mma,
        )
        return None

    # Entering extreme zone — fire alert
    alert = {
        "date": current_date,
        "event": f"MSASI_ENTER_{current_zone}",
        "current_zone": current_zone,
        "previous_zone": prev_zone,
        "msasi_1mma": current_msasi_1mma,
        "previous_msasi_1mma": previous.get("msasi_1mma"),
        "threshold": (
            OVERSOLD_THRESHOLD if current_zone == "OVERSOLD"
            else OVERBOUGHT_THRESHOLD
        ),
        "suggested_action": (
            "LONG CSI300 (70-day hold)" if current_zone == "OVERSOLD"
            else "SHORT CSI300 (70-day hold)"
        ),
    }

    logger.warning(
        "🔔 SENTIMENT EXTREME: %s — MSASI 1MMA=%.1f%% (threshold=%d%%)",
        current_zone, current_msasi_1mma,
        alert["threshold"],
    )

    return alert


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_sentiment_alert(alert: dict) -> str:
    """Format a human-readable sentiment alert message.

    Parameters
    ----------
    alert : dict
        Alert info from check_sentiment_extreme().

    Returns
    -------
    str
        Formatted alert string.
    """
    lines = [
        "",
        "=" * 60,
        f"🔔 MSASI SENTIMENT EXTREME — {alert['date']}",
        "=" * 60,
        f"  Event:     {alert['event']}",
        f"  Zone:      {alert['previous_zone']} → {alert['current_zone']}",
        f"  MSASI 1MMA: {alert['msasi_1mma']:.1f}%",
        f"  Threshold: {alert['threshold']}%",
        f"  Action:    {alert['suggested_action']}",
        "",
        "  Historical context:",
    ]

    if alert["current_zone"] == "OVERSOLD":
        lines.append(
            "    MSASI ≤ 25%: historically marks market bottoms."
        )
        lines.append(
            "    Paper reports 67% hit rate, 91% cumulative return (LONG)."
        )
    else:
        lines.append(
            "    MSASI ≥ 75%: historically marks market overheating."
        )
        lines.append(
            "    Paper reports 60% hit rate, 31% cumulative return (SHORT)."
        )

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_alert_log(alert: dict) -> Path:
    """Append an alert to the JSONL log file.

    Returns
    -------
    Path
        Path to the log file.
    """
    log_path = _alert_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {**alert, "date": str(alert["date"])}
    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    logger.info("Sentiment alert logged to %s", log_path)
    return log_path
