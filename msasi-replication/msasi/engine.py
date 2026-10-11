"""MSASI 2.0 — Core engine.

Implements the 4-step methodology from the Morgan Stanley paper:
1. 100-day rolling min-max normalization
2. Expanding-window R²-weighted aggregation
3. MSASI raw + MSASI 1MMA (20-day moving average)
4. Extreme signal generation with cooldown
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    COOLDOWN_DAYS,
    HOLDING_PERIOD,
    INDICATOR_MAP,
    INDICATOR_KEYS,
    INDICATOR_SPECS,
    MMA_WINDOW,
    NORM_WINDOW,
    OVERBOUGHT_THRESHOLD,
    OVERSOLD_THRESHOLD,
    WEIGHT_ANCHOR_START,
    WEIGHT_FWD_RETURN_DAYS,
    WEIGHT_MIN_HISTORY_DAYS,
    N_INDICATORS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MSASIResult:
    """Container for the full MSASI pipeline output."""

    msasi_daily: pd.DataFrame
    """DataFrame with [date, msasi_raw, msasi_1mma]."""

    signals: pd.DataFrame
    """DataFrame with [date, signal_type] where signal_type ∈ {LONG, SHORT}."""

    weights: pd.Series
    """Final indicator weights (indicator key → weight)."""

    component_raw: pd.DataFrame
    """DataFrame of raw indicator values."""

    component_normalized: pd.DataFrame
    """DataFrame of all 100-day normalized components for diagnostics."""

    weight_history: Optional[pd.DataFrame] = None
    """Time-varying weights if using expanding window."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MSASIEngine:
    """Morgan Stanley A-Share Sentiment Indicator (MSASI) 2.0 engine.

    Usage::

        from msasi.engine import MSASIEngine
        from msasi.data.readers import load_all_indicators, load_allshare_prices

        raw = load_all_indicators(start_date="2013-01-01")
        csi = load_allshare_prices()
        engine = MSASIEngine()
        result = engine.run(raw, csi)
    """

    def __init__(
        self,
        norm_window: int = NORM_WINDOW,
        mma_window: int = MMA_WINDOW,
        overbought: float = OVERBOUGHT_THRESHOLD,
        oversold: float = OVERSOLD_THRESHOLD,
        cooldown_days: int = COOLDOWN_DAYS,
        weight_anchor: str = WEIGHT_ANCHOR_START,
        weight_fwd_days: int = WEIGHT_FWD_RETURN_DAYS,
        weight_min_history: int = WEIGHT_MIN_HISTORY_DAYS,
    ):
        self.norm_window = norm_window
        self.mma_window = mma_window
        self.overbought = overbought
        self.oversold = oversold
        self.cooldown_days = cooldown_days
        self.weight_anchor = pd.Timestamp(weight_anchor)
        self.weight_fwd_days = weight_fwd_days
        self.weight_min_history = weight_min_history

    # ----- Step 1: Normalization -----

    def normalize(self, raw_indicators: pd.DataFrame) -> pd.DataFrame:
        """Apply 100-day rolling min-max normalization to each indicator.

        Formula: (value - rolling_min) / (rolling_max - rolling_min) * 100

        Parameters
        ----------
        raw_indicators : pd.DataFrame
            Must contain 'date' column and indicator key columns.

        Returns
        -------
        pd.DataFrame
            Same shape, with normalized values in [0, 100].
        """
        result = raw_indicators[["date"]].copy()
        indicator_cols = [k for k in INDICATOR_KEYS if k in raw_indicators.columns]

        for col in indicator_cols:
            series = raw_indicators[col].astype(float)
            roll_min = series.rolling(window=self.norm_window, min_periods=1).min()
            roll_max = series.rolling(window=self.norm_window, min_periods=1).max()
            denom = roll_max - roll_min
            # Avoid division by zero: when range is 0, normalized = 50 (neutral)
            normalized = ((series - roll_min) / denom.replace(0, np.nan)) * 100.0
            result[col] = normalized

        logger.info(
            "Normalized %d indicators over %d-day rolling window",
            len(indicator_cols),
            self.norm_window,
        )
        return result

    # ----- Step 2: Weight computation -----

    def compute_weights(
        self,
        normalized: pd.DataFrame,
        allshare_returns: pd.Series,
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        """Compute R²-weighted indicator weights using expanding window.

        For each indicator, regress its normalized value against CSI300
        forward N-day returns and compute R². Multiply by penalty factor
        for monthly/short-history indicators.

        Parameters
        ----------
        normalized : pd.DataFrame
            Normalized indicator values with 'date' column.
        allshare_returns : pd.Series
            CSI300 forward returns aligned to the same date index.
        as_of_date : pd.Timestamp, optional
            Compute weights using only data up to this date.
            If None, uses all available data.

        Returns
        -------
        pd.Series
            Indicator key → weight (normalized to sum to 1).
        """
        indicator_cols = [k for k in INDICATOR_KEYS if k in normalized.columns]

        # Align data
        df = normalized.copy()
        df["fwd_return"] = allshare_returns.values if len(allshare_returns) == len(df) else np.nan

        # Filter dates
        mask = df["date"] >= self.weight_anchor
        if as_of_date is not None:
            mask &= df["date"] <= as_of_date
        df = df[mask].copy()

        if len(df) < self.weight_min_history:
            # Not enough data — use equal weights
            logger.warning(
                "Only %d days available for weight estimation (need %d). Using equal weights.",
                len(df), self.weight_min_history,
            )
            w = pd.Series(1.0, index=indicator_cols)
            return w / w.sum()

        weights = {}
        for col in indicator_cols:
            valid = df[[col, "fwd_return"]].dropna()
            if len(valid) < 30:
                weights[col] = 0.0
                continue

            x = valid[col].values
            y = valid["fwd_return"].values
            # R² = correlation^2
            corr = np.corrcoef(x, y)
            if corr.shape == (2, 2) and np.isfinite(corr[0, 1]):
                r2 = corr[0, 1] ** 2
            else:
                r2 = 0.0

            # Apply penalty from spec
            spec = INDICATOR_MAP.get(col)
            penalty = spec.weight_penalty if spec else 1.0
            weights[col] = r2 * penalty

        w = pd.Series(weights)
        total = w.sum()
        if total > 0:
            w = w / total
        else:
            w = pd.Series(1.0 / len(indicator_cols), index=indicator_cols)

        logger.info("Computed weights (top 5): %s", w.nlargest(5).to_dict())
        return w

    def compute_expanding_weights(
        self,
        normalized: pd.DataFrame,
        allshare_returns: pd.Series,
    ) -> pd.DataFrame:
        """Compute time-varying weights using expanding window.

        Re-computes weights at each month-end using all data available
        up to that point.

        Returns
        -------
        pd.DataFrame
            Columns: [date] + indicator keys with weight values.
        """
        dates = normalized["date"]
        indicator_cols = [k for k in INDICATOR_KEYS if k in normalized.columns]

        # Compute weights at monthly intervals
        month_ends = dates.dt.to_period("M").drop_duplicates()
        weight_records = []

        for period in month_ends:
            # Normalize to midnight — to_timestamp(how="end") returns 23:59:59.999
            as_of = period.to_timestamp(how="end").normalize()
            if as_of < self.weight_anchor + pd.Timedelta(days=self.weight_min_history):
                # Not enough history yet
                w = pd.Series(1.0, index=indicator_cols)
                w = w / w.sum()
            else:
                w = self.compute_weights(normalized, allshare_returns, as_of_date=as_of)

            record = {"date": as_of}
            for col in indicator_cols:
                record[col] = w.get(col, 0.0)
            weight_records.append(record)

        if not weight_records:
            return pd.DataFrame()

        wdf = pd.DataFrame(weight_records)
        # Forward-fill monthly weights to daily trading calendar
        daily_idx = pd.bdate_range(dates.min(), dates.max())
        wdf = wdf.set_index("date").reindex(daily_idx).ffill().bfill()
        wdf.index.name = "date"
        return wdf.reset_index()

    # ----- Step 3: Aggregation -----

    def aggregate(
        self,
        normalized: pd.DataFrame,
        weights: pd.Series | pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate normalized indicators into MSASI raw and 1MMA.

        Parameters
        ----------
        normalized : pd.DataFrame
            Normalized indicator values with 'date' column.
        weights : pd.Series or pd.DataFrame
            If Series: static weights. If DataFrame: time-varying weights with 'date'.

        Returns
        -------
        pd.DataFrame
            Columns: [date, msasi_raw, msasi_1mma]
        """
        indicator_cols = [k for k in INDICATOR_KEYS if k in normalized.columns]
        result = normalized[["date"]].copy()

        if isinstance(weights, pd.Series):
            # Static weights
            vals = normalized[indicator_cols].values  # (T, N)
            w = np.array([weights.get(c, 0.0) for c in indicator_cols])  # (N,)
            # Handle NaN: use available-indicator weighted average
            mask_valid = ~np.isnan(vals)
            w_expanded = np.tile(w, (vals.shape[0], 1))  # (T, N)
            w_masked = np.where(mask_valid, w_expanded, 0.0)
            w_sum = w_masked.sum(axis=1, keepdims=True)
            w_sum = np.where(w_sum > 0, w_sum, np.nan)
            w_normed = w_masked / w_sum
            vals_filled = np.where(mask_valid, vals, 0.0)
            msasi_raw = (vals_filled * w_normed).sum(axis=1)
        else:
            # Time-varying weights — vectorized
            weight_df = weights.copy()
            if "date" not in weight_df.columns:
                weight_df = weight_df.reset_index()

            # Use merge_asof for robust temporal alignment (forward-fill)
            norm_with_date = normalized[["date"]].copy()
            norm_with_date = norm_with_date.sort_values("date")
            weight_df = weight_df.sort_values("date")
            w_aligned = pd.merge_asof(
                norm_with_date, weight_df, on="date", direction="backward"
            )
            # For dates before any weight data, use first available weights
            for col in indicator_cols:
                if col in w_aligned.columns:
                    w_aligned[col] = w_aligned[col].bfill()

            # Extract aligned arrays
            vals = normalized[indicator_cols].astype(float).values  # (T, N)
            w_mat = w_aligned[indicator_cols].astype(float).values  # (T, N)

            # NaN-aware weighted average (same logic, vectorized)
            mask_valid = ~np.isnan(vals)
            w_masked = np.where(mask_valid, w_mat, 0.0)
            w_sum = w_masked.sum(axis=1, keepdims=True)
            w_sum = np.where(w_sum > 0, w_sum, np.nan)
            w_normed = w_masked / w_sum
            vals_filled = np.where(mask_valid, vals, 0.0)
            msasi_raw = (vals_filled * w_normed).sum(axis=1)

        result["msasi_raw"] = msasi_raw
        result["msasi_1mma"] = result["msasi_raw"].rolling(
            window=self.mma_window, min_periods=1
        ).mean()

        logger.info(
            "MSASI computed: raw range [%.1f, %.1f], 1MMA range [%.1f, %.1f]",
            result["msasi_raw"].min(),
            result["msasi_raw"].max(),
            result["msasi_1mma"].min(),
            result["msasi_1mma"].max(),
        )
        return result

    # ----- Step 4: Signal generation -----

    def generate_signals(
        self,
        msasi_1mma: pd.Series,
        dates: pd.Series,
        holding_period: int = HOLDING_PERIOD,
    ) -> pd.DataFrame:
        """Detect sentiment extremes with non-overlapping trade logic.

        Rules (from paper):
        - LONG when MSASI 1MMA ≤ oversold_threshold (22%)
        - SHORT when MSASI 1MMA ≥ overbought_threshold (78%)
        - Only the FIRST crossing into the extreme zone fires a signal.
          The indicator must EXIT the extreme zone AND any active trade
          of that type must have closed before a new signal fires.
        - Holding period is 70 trading days — no new same-type signal
          while a position is active.

        Parameters
        ----------
        msasi_1mma : pd.Series
            The MSASI 1MMA values.
        dates : pd.Series
            Corresponding dates.
        holding_period : int
            Trading days per position (used to block overlapping trades).

        Returns
        -------
        pd.DataFrame
            Columns: [date, signal_type] where signal_type ∈ {'LONG', 'SHORT'}.
        """
        signals = []
        in_oversold = False
        in_overbought = False
        last_long_exit_idx = -1   # Index when the last LONG position closes
        last_short_exit_idx = -1  # Index when the last SHORT position closes

        for i in range(len(msasi_1mma)):
            val = msasi_1mma.iloc[i]
            if np.isnan(val):
                continue

            # Long signal: first entry into oversold zone, no active long trade
            if val <= self.oversold:
                if not in_oversold and i > last_long_exit_idx:
                    signals.append({
                        "date": dates.iloc[i],
                        "signal_type": "LONG",
                        "msasi_1mma": val,
                    })
                    in_oversold = True
                    last_long_exit_idx = i + holding_period + 1  # T+1 entry + holding
            else:
                in_oversold = False

            # Short signal: first entry into overbought zone, no active short trade
            if val >= self.overbought:
                if not in_overbought and i > last_short_exit_idx:
                    signals.append({
                        "date": dates.iloc[i],
                        "signal_type": "SHORT",
                        "msasi_1mma": val,
                    })
                    in_overbought = True
                    last_short_exit_idx = i + holding_period + 1
            else:
                in_overbought = False

        result = pd.DataFrame(signals)
        if result.empty:
            result = pd.DataFrame(columns=["date", "signal_type", "msasi_1mma"])

        logger.info(
            "Generated %d signals (%d LONG, %d SHORT)",
            len(result),
            (result["signal_type"] == "LONG").sum() if not result.empty else 0,
            (result["signal_type"] == "SHORT").sum() if not result.empty else 0,
        )
        return result

    # ----- Full pipeline -----

    def run(
        self,
        raw_indicators: pd.DataFrame,
        allshare_prices: pd.DataFrame,
        use_expanding_weights: bool = True,
    ) -> MSASIResult:
        """Execute the full MSASI 2.0 pipeline.

        Parameters
        ----------
        raw_indicators : pd.DataFrame
            Output of ``load_all_indicators()``.
        allshare_prices : pd.DataFrame
            CSI300 daily prices with [date, close] columns.
        use_expanding_weights : bool
            If True, re-compute weights monthly using expanding window.
            If False, compute weights once using all available data.

        Returns
        -------
        MSASIResult
        """
        logger.info("=" * 60)
        logger.info("MSASI 2.0 Pipeline — Starting")
        logger.info("=" * 60)

        # Ensure alignment
        raw = raw_indicators.copy()
        raw["date"] = pd.to_datetime(raw["date"])
        csi = allshare_prices.copy()
        csi["date"] = pd.to_datetime(csi["date"])

        # Compute CSI300 forward returns (for R² weighting)
        csi = csi.sort_values("date").reset_index(drop=True)
        csi["fwd_return"] = csi["close"].pct_change(self.weight_fwd_days).shift(-self.weight_fwd_days)

        # Align returns to indicator dates
        returns_map = csi.set_index("date")["fwd_return"]
        raw_aligned = raw.copy()
        raw_aligned["fwd_return"] = raw_aligned["date"].map(returns_map)

        # Step 1: Normalize
        normalized = self.normalize(raw)

        # Step 2: Compute weights
        allshare_fwd = raw_aligned["fwd_return"]
        if use_expanding_weights:
            weight_history = self.compute_expanding_weights(normalized, allshare_fwd)
            # Use latest weights for summary
            indicator_cols = [k for k in INDICATOR_KEYS if k in normalized.columns]
            if not weight_history.empty:
                final_weights = weight_history[indicator_cols].iloc[-1]
            else:
                final_weights = pd.Series(1.0 / len(indicator_cols), index=indicator_cols)
        else:
            final_weights = self.compute_weights(normalized, allshare_fwd)
            weight_history = None

        # Step 3: Aggregate
        if use_expanding_weights and weight_history is not None and not weight_history.empty:
            msasi_df = self.aggregate(normalized, weight_history)
        else:
            msasi_df = self.aggregate(normalized, final_weights)

        # Step 4: Generate signals
        signals = self.generate_signals(msasi_df["msasi_1mma"], msasi_df["date"])

        logger.info("=" * 60)
        logger.info("MSASI 2.0 Pipeline — Complete")
        logger.info("=" * 60)

        return MSASIResult(
            msasi_daily=msasi_df,
            signals=signals,
            weights=final_weights,
            component_raw=raw,
            component_normalized=normalized,
            weight_history=weight_history,
        )
