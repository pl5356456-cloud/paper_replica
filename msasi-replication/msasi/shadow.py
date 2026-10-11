"""MSASI shadow — PCA weights + build/trim position rule.

Runs beside the production R²-weighted MSASI without changing it. Why it exists
(walk-forward study, 2026-09-25, eval 2017→): the R² weights ignore the sign of
each indicator's link to forward returns, so pro-cyclical indicators such as
futures_basis (~0.37 weight) cancel the contrarian turnover signal and the
composite has no skill (IC60 −0.03). The first principal component of the
normalized indicators — the common "market heat" factor — never looks at
returns and scored IC60 −0.29 (Newey-West t −3.0), stable in both halves.

Pipeline
--------
1. Reuse the engine's 100-day min-max scores; drop ``foreign_passive_flow_1mma``
   (it is a 20-day mean of ``northbound_turnover``, not a separate feed).
2. At each month-end, fit PC1 on all scores up to that date (pairwise
   covariance, expanding from 2014). Keep positive loadings only; indicators
   that load negatively move against the heat factor and are left out. The
   weights apply from the next trading day. Equal weights before the first fit.
3. Composite → 20-day mean → point-in-time percentile against its own history
   (≥ 500 observations). A PCA composite does not spread over 0-100 the way the
   R² one does, so zones are percentiles, not 25/75.
4. Zones: ≤ 15th pct OVERSOLD, ≥ 85th pct OVERBOUGHT.
5. Position (one-sided — the overbought side does not justify a short):
   base 50%; build to 100% on entering OVERSOLD and hold until the percentile
   crosses back up through 50; cut to 0% (flat) on entering OVERBOUGHT and hold
   until it crosses back down through 50. The signal expires when sentiment has
   normalised. Tested against the paper's fixed 70-day hold and against holding
   until the next signal (2026-09-25, 2017→): Sharpe 0.54 vs 0.39 vs 0.26,
   better at every zone width and in both halves. It is the only rule that
   beat a circular-shift timing placebo (p = 0.035; the fixed hold p = 0.24).

Outputs (``cn/equities/sentiment/msasi/``): ``shadow_pca_daily.parquet``,
``shadow_pca_events.parquet`` (position changes), ``shadow_pca_weights.parquet`` (monthly history)
and ``shadow_pca_backtest.parquet`` (rule vs buy & hold vs constant 50%).
"""
from __future__ import annotations

import bisect
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXCLUDE = ("foreign_passive_flow_1mma",)
FIT_START = "2014-01-01"
MIN_FIT_ROWS = 250
MIN_PCT_HISTORY = 500
MMA_WINDOW = 20
OVERSOLD_PCT = 15.0
OVERBOUGHT_PCT = 85.0
EXIT_PCT = 50.0
BASE_POS, BUILD_POS, TRIM_POS = 0.5, 1.0, 0.0
COST_PER_TURNOVER = 0.001  # 10 bp per unit of exposure changed
EVAL_START = "2017-01-01"


def pc1_weights(Z: pd.DataFrame) -> pd.Series:
    """Positive PC1 loadings of the standardized scores, normalized to sum 1."""
    cov = ((Z - Z.mean()) / Z.std()).cov(min_periods=MIN_FIT_ROWS)
    cov = cov.dropna(how="all").dropna(axis=1, how="all")
    cols = cov.index[cov.notna().all(axis=1)]
    cov = cov.loc[cols, cols]
    if len(cols) < 2:
        return pd.Series(1.0 / Z.shape[1], index=Z.columns)
    vals, vecs = np.linalg.eigh(cov.values)
    v = pd.Series(vecs[:, -1], index=cols)
    v *= np.sign(v.sum())
    w = v.clip(lower=0).reindex(Z.columns).fillna(0.0)
    return w / w.sum()


def fit_weight_history(norm: pd.DataFrame) -> pd.DataFrame:
    """Monthly PC1 weights, each fitted only on scores up to that month-end."""
    Z = norm.set_index("date")
    Z = Z[Z.index >= FIT_START]
    ends = Z.index.to_series().groupby(Z.index.to_period("M")).max()
    rows = {}
    for me in ends:
        hist = Z[Z.index <= me]
        if len(hist) < MIN_FIT_ROWS:
            continue
        rows[me] = pc1_weights(hist)
    W = pd.DataFrame(rows).T
    W.index.name = "date"
    return W


def composite(norm: pd.DataFrame, W: pd.DataFrame) -> pd.Series:
    """NaN-aware weighted mean; month-end weights apply from the next day."""
    Z = norm.set_index("date")
    Wd = W.reindex(Z.index, method="ffill").shift(1)
    first_fit = Wd.notna().any(axis=1)
    Wd[~first_fit] = 1.0 / Z.shape[1]  # equal weights before the first fit
    A = Wd[Z.columns].where(Z.notna(), 0.0)
    return (Z.fillna(0.0) * A).sum(axis=1) / A.sum(axis=1).replace(0, np.nan)


def pit_percentile(s: pd.Series) -> pd.Series:
    """Percentile of each value within all values up to and including it."""
    out = np.full(len(s), np.nan)
    hist: list[float] = []
    for i, v in enumerate(s.values):
        if np.isnan(v):
            continue
        bisect.insort(hist, v)
        if len(hist) >= MIN_PCT_HISTORY:
            out[i] = bisect.bisect_right(hist, v) / len(hist) * 100
    return pd.Series(out, index=s.index)


def position_path(pct: pd.Series) -> pd.Series:
    """Target exposure: build/trim on zone entry, back to base once the percentile crosses 50."""
    pos = np.full(len(pct), BASE_POS)
    state = BASE_POS
    for i, v in enumerate(pct.values):
        if not np.isnan(v):
            if v <= OVERSOLD_PCT:
                state = BUILD_POS
            elif v >= OVERBOUGHT_PCT:
                state = TRIM_POS
            elif (state == BUILD_POS and v >= EXIT_PCT) or (state == TRIM_POS and v <= EXIT_PCT):
                state = BASE_POS
        pos[i] = state
    return pd.Series(pos, index=pct.index)


def position_events(pos: pd.Series, pct: pd.Series, close: pd.Series) -> pd.DataFrame:
    """One row per position change, with how long it was held and the index return meanwhile."""
    change = pos.ne(pos.shift()) & pos.shift().notna()
    dates = pos.index[change]
    action = {BUILD_POS: "BUILD", TRIM_POS: "TRIM", BASE_POS: "EXIT"}
    rows = []
    for i, d in enumerate(dates):
        nxt = dates[i + 1] if i + 1 < len(dates) else None
        held = pos.loc[d:nxt].iloc[:-1] if nxt is not None else pos.loc[d:]
        rows.append({
            "date": d, "action": action[pos[d]], "position": float(pos[d]), "pct": float(pct[d]),
            "days_held": len(held),
            "index_ret_held": (close[nxt] / close[d] - 1) if nxt is not None else np.nan,
            "open": nxt is None,
        })
    return pd.DataFrame(rows, columns=["date", "action", "position", "pct", "days_held", "index_ret_held", "open"])


def backtest(close: pd.Series, pos: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Next-day execution, 10 bp per unit turnover. Returns equity curves and stats."""
    ret = close.pct_change()
    books = {
        "Build/trim rule": pos,
        "Buy & hold": pd.Series(1.0, index=close.index),
        "Constant 50%": pd.Series(BASE_POS, index=close.index),
    }
    curves, stats = {}, []
    for name, p in books.items():
        p = p.reindex(close.index).ffill().fillna(BASE_POS)
        pnl = p.shift(1) * ret - p.diff().abs().fillna(0) * COST_PER_TURNOVER
        pnl = pnl[pnl.index >= EVAL_START].fillna(0)
        eq = (1 + pnl).cumprod()
        curves[name] = eq
        yrs = len(pnl) / 244
        stats.append({
            "strategy": name,
            "cagr": eq.iloc[-1] ** (1 / yrs) - 1,
            "vol": pnl.std() * np.sqrt(244),
            "sharpe": pnl.mean() / pnl.std() * np.sqrt(244),
            "max_drawdown": (eq / eq.cummax() - 1).min(),
            "avg_exposure": p[p.index >= EVAL_START].mean(),
        })
    eq = pd.DataFrame(curves)
    eq.index.name = "date"
    return eq.reset_index(), pd.DataFrame(stats)


def run_shadow(normalized: pd.DataFrame, allshare_prices: pd.DataFrame, out_dir: Path | None = None) -> dict:
    """Build the shadow series from the engine's normalized components and save it."""
    cols = [c for c in normalized.columns if c != "date" and c not in EXCLUDE]
    norm = normalized[["date"] + cols].copy()
    norm["date"] = pd.to_datetime(norm["date"])
    norm = norm.sort_values("date").reset_index(drop=True)

    W = fit_weight_history(norm)
    comp = composite(norm, W)
    mma = comp.rolling(MMA_WINDOW, min_periods=1).mean()
    pct = pit_percentile(mma)
    pos = position_path(pct)

    px = allshare_prices[["date", "close"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    close = px.set_index("date")["close"].astype(float).reindex(pct.index).ffill()

    daily = pd.DataFrame({
        "composite": comp, "composite_1mma": mma, "percentile": pct,
        "zone": np.where(pct <= OVERSOLD_PCT, "OVERSOLD", np.where(pct >= OVERBOUGHT_PCT, "OVERBOUGHT", "NEUTRAL")),
        "position": pos, "close": close,
    })
    daily.loc[daily["percentile"].isna(), "zone"] = "UNKNOWN"
    daily.index.name = "date"
    daily = daily.reset_index()

    events = position_events(pos, pct, close)
    equity, stats = backtest(close, pos)

    wh = W.reset_index().melt(id_vars="date", var_name="indicator", value_name="weight")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(out_dir / "shadow_pca_daily.parquet", index=False)
        events.to_parquet(out_dir / "shadow_pca_events.parquet", index=False)
        wh.to_parquet(out_dir / "shadow_pca_weights.parquet", index=False)
        stats.assign(eval_start=EVAL_START).to_parquet(out_dir / "shadow_pca_backtest.parquet", index=False)
        equity.to_parquet(out_dir / "shadow_pca_equity.parquet", index=False)

    last = daily.iloc[-1]
    logger.info("MSASI shadow (PCA): %s pct=%.1f zone=%s position=%.0f%%",
                last["date"].date(), last["percentile"], last["zone"], last["position"] * 100)
    return {"daily": daily, "events": events, "weights": W, "backtest": stats, "equity": equity}
