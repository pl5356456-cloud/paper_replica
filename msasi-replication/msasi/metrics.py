"""Performance metrics used by the backtest.

Only ``compute_perf_stats`` / ``PerfStats`` and the helpers they call, copied
as written from the original project's shared metrics module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

ANNUALIZATION = 252  # trading days per year


def equity_curve(daily_returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compute cumulative equity curve from daily returns."""
    r = pd.to_numeric(daily_returns, errors="coerce").fillna(0.0)
    return initial * (1.0 + r).cumprod()


def max_drawdown(daily_returns: pd.Series) -> float:
    """Maximum drawdown from daily return series. Returns negative value."""
    eq = equity_curve(daily_returns)
    if eq.empty:
        return np.nan
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def cagr(
    daily_returns: pd.Series,
    ann: int = ANNUALIZATION,
) -> float:
    """Compound annual growth rate."""
    r = pd.to_numeric(daily_returns, errors="coerce").dropna()
    n = len(r)
    if n == 0:
        return np.nan
    eq = (1.0 + r).prod()
    if eq <= 0:
        return np.nan
    years = n / ann
    return float(eq ** (1.0 / years) - 1.0) if years > 0 else np.nan


def ann_vol(
    daily_returns: pd.Series,
    ann: int = ANNUALIZATION,
    ddof: int = 1,
) -> float:
    """Annualized volatility."""
    r = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=ddof) * math.sqrt(ann))


def sharpe(
    daily_returns: pd.Series,
    ann: int = ANNUALIZATION,
    rf: float = 0.0,
    ddof: int = 1,
) -> float:
    """Annualized Sharpe ratio (excess of rf)."""
    r = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if len(r) < 2:
        return np.nan
    mu = float(r.mean())
    sd = float(r.std(ddof=ddof))
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return float(((mu - rf / ann) / sd) * math.sqrt(ann))


def calmar(
    daily_returns: pd.Series,
    ann: int = ANNUALIZATION,
) -> float:
    """Calmar ratio = CAGR / |MaxDD|."""
    c = cagr(daily_returns, ann=ann)
    mdd = max_drawdown(daily_returns)
    if not np.isfinite(c) or not np.isfinite(mdd) or mdd >= 0:
        return np.nan
    return float(c / abs(mdd))


def cvar(
    daily_returns: pd.Series,
    alpha: float = 0.05,
) -> float:
    """Conditional Value-at-Risk (expected shortfall) at given alpha."""
    r = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if r.empty:
        return np.nan
    q = float(np.quantile(r.values, alpha))
    tail = r[r <= q]
    if tail.empty:
        return float(q)
    return float(tail.mean())


def hit_rate(daily_returns: pd.Series) -> float:
    """Fraction of positive-return days."""
    r = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if r.empty:
        return np.nan
    return float((r > 0).mean())


# ---------------------------------------------------------------------------
# PerfStats — the single canonical dataclass
# ---------------------------------------------------------------------------

@dataclass
class PerfStats:
    """Unified performance statistics."""
    cagr: float = np.nan
    ann_vol: float = np.nan
    sharpe: float = np.nan
    max_drawdown: float = np.nan
    calmar: float = np.nan
    win_rate: float = np.nan
    avg_daily_return: float = np.nan
    worst_day: float = np.nan
    cvar_95: float = np.nan
    n_days: int = 0
    avg_daily_turnover: float = np.nan
    total_turnover: float = np.nan

    def to_dict(self) -> Dict[str, float]:
        return {
            "cagr": self.cagr,
            "ann_vol": self.ann_vol,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "win_rate": self.win_rate,
            "avg_daily_return": self.avg_daily_return,
            "worst_day": self.worst_day,
            "cvar_95": self.cvar_95,
            "n_days": float(self.n_days),
            "avg_daily_turnover": self.avg_daily_turnover,
            "total_turnover": self.total_turnover,
        }

    def to_series(self) -> pd.Series:
        return pd.Series(self.to_dict())


def compute_perf_stats(
    daily_returns: pd.Series,
    *,
    daily_turnover: Optional[pd.Series] = None,
    ann: int = ANNUALIZATION,
) -> PerfStats:
    """Compute full performance statistics from a daily return series.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily strategy returns.
    daily_turnover : pd.Series, optional
        Daily turnover values (0-1 scale). If None, turnover stats are NaN.
    ann : int
        Annualization factor (trading days per year).
    """
    r = pd.to_numeric(daily_returns, errors="coerce").dropna()
    n = int(len(r))
    if n == 0:
        return PerfStats(n_days=0)

    _cagr = cagr(r, ann=ann)
    _ann_vol = ann_vol(r, ann=ann)
    _sharpe = sharpe(r, ann=ann)
    _calmar = calmar(r, ann=ann)
    _mdd = max_drawdown(r)
    _win = hit_rate(r)
    _avg = float(r.mean())
    _worst = float(r.min())
    _cvar95 = cvar(r, alpha=0.05)

    # Turnover stats
    if daily_turnover is not None:
        t = pd.to_numeric(daily_turnover, errors="coerce").reindex(r.index).fillna(0.0)
        _avg_to = float(t.mean())
        _total_to = float(t.sum())
    else:
        _avg_to = np.nan
        _total_to = np.nan

    return PerfStats(
        cagr=_cagr if np.isfinite(_cagr) else np.nan,
        ann_vol=_ann_vol if np.isfinite(_ann_vol) else np.nan,
        sharpe=_sharpe if np.isfinite(_sharpe) else np.nan,
        max_drawdown=_mdd if np.isfinite(_mdd) else np.nan,
        calmar=_calmar if np.isfinite(_calmar) else np.nan,
        win_rate=_win if np.isfinite(_win) else np.nan,
        avg_daily_return=_avg if np.isfinite(_avg) else np.nan,
        worst_day=_worst if np.isfinite(_worst) else np.nan,
        cvar_95=_cvar95 if np.isfinite(_cvar95) else np.nan,
        n_days=n,
        avg_daily_turnover=_avg_to,
        total_turnover=_total_to,
    )
