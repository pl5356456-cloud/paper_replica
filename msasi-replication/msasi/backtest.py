"""MSASI 2.0 — Event-driven backtest.

Implements the paper's trading strategy:
- LONG CSI300 for 70 trading days when MSASI 1MMA ≤ 22% (first touch in 20-day window)
- SHORT CSI300 for 70 trading days when MSASI 1MMA ≥ 78% (first touch in 20-day window)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import HOLDING_PERIOD, OVERBOUGHT_THRESHOLD, OVERSOLD_THRESHOLD
from .metrics import compute_perf_stats, PerfStats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """A single trade from the MSASI strategy."""
    signal_type: str        # "LONG" or "SHORT"
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    holding_days: int
    raw_return: float       # (exit - entry) / entry for LONG, negative for SHORT
    strategy_return: float  # raw_return for LONG, -raw_return for SHORT
    msasi_at_signal: float


@dataclass
class MSASIBacktestResult:
    """Backtest results container."""
    trades: List[TradeRecord]
    long_trades: List[TradeRecord]
    short_trades: List[TradeRecord]
    summary: dict
    long_summary: dict
    short_summary: dict
    daily_pnl: pd.DataFrame   # Reconstructed daily P&L
    perf_stats: Optional[PerfStats] = None


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_msasi_backtest(
    signals: pd.DataFrame,
    allshare_prices: pd.DataFrame,
    holding_period: int = HOLDING_PERIOD,
    trading_cost_bps: float = 10.0,
    short_borrow_bps: float = 300.0,
) -> MSASIBacktestResult:
    """Execute the MSASI event-driven backtest.

    Parameters
    ----------
    signals : pd.DataFrame
        Output of ``MSASIEngine.generate_signals()`` with [date, signal_type, msasi_1mma].
    allshare_prices : pd.DataFrame
        CSI300 daily prices with [date, open, close].
    holding_period : int
        Number of trading days to hold each trade (default 70).
    trading_cost_bps : float
        Round-trip execution slippage and commissions in basis points.
    short_borrow_bps : float
        Annualized short borrowing cost in basis points.

    Returns
    -------
    MSASIBacktestResult
    """
    if signals.empty:
        logger.warning("No signals to backtest")
        return MSASIBacktestResult(
            trades=[], long_trades=[], short_trades=[],
            summary={}, long_summary={}, short_summary={},
            daily_pnl=pd.DataFrame(),
        )

    # Prepare price lookup
    prices = allshare_prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date").reset_index(drop=True)
    price_dates = prices["date"].values
    price_close = prices["close"].values
    price_open = prices["open"].values if "open" in prices.columns else price_close

    # Create date→index mapping
    date_to_idx = {d: i for i, d in enumerate(price_dates)}

    trades: List[TradeRecord] = []

    for _, sig in signals.iterrows():
        sig_date = pd.Timestamp(sig["date"])
        sig_type = sig["signal_type"]
        msasi_val = sig.get("msasi_1mma", np.nan)

        # Find entry: T+1 open after signal date
        sig_idx = date_to_idx.get(sig_date)
        if sig_idx is None:
            # Find nearest trading date
            valid = [i for i, d in enumerate(price_dates) if d >= sig_date]
            if not valid:
                continue
            sig_idx = valid[0]

        entry_idx = sig_idx + 1
        if entry_idx >= len(price_dates):
            continue

        # Exit: entry + holding_period trading days
        exit_idx = entry_idx + holding_period
        if exit_idx >= len(price_dates):
            exit_idx = len(price_dates) - 1

        entry_date = pd.Timestamp(price_dates[entry_idx])
        exit_date = pd.Timestamp(price_dates[exit_idx])
        entry_price = float(price_open[entry_idx])
        exit_price = float(price_close[exit_idx])
        actual_holding = exit_idx - entry_idx

        # Cost modeling
        exec_cost = trading_cost_bps / 10000.0
        borrow_cost = 0.0
        if sig_type == "SHORT":
            borrow_cost = (short_borrow_bps / 10000.0) * (actual_holding / 252.0)

        # Raw return
        raw_ret = (exit_price - entry_price) / entry_price
        
        # Strategy return: positive for correct direction, minus costs
        if sig_type == "LONG":
            strat_ret = raw_ret - exec_cost
        else:
            strat_ret = -raw_ret - exec_cost - borrow_cost

        trade = TradeRecord(
            signal_type=sig_type,
            signal_date=sig_date,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            holding_days=actual_holding,
            raw_return=raw_ret,
            strategy_return=strat_ret,
            msasi_at_signal=msasi_val,
        )
        trades.append(trade)

    # Split by direction
    long_trades = [t for t in trades if t.signal_type == "LONG"]
    short_trades = [t for t in trades if t.signal_type == "SHORT"]

    # Summaries
    summary = _compute_trade_summary(trades, "ALL")
    long_summary = _compute_trade_summary(long_trades, "LONG")
    short_summary = _compute_trade_summary(short_trades, "SHORT")

    # Build daily P&L series (for canonical perf stats)
    daily_pnl = _build_daily_pnl(trades, prices, holding_period, trading_cost_bps, short_borrow_bps)
    perf = None
    if not daily_pnl.empty and "strategy_ret" in daily_pnl.columns:
        perf = compute_perf_stats(daily_pnl["strategy_ret"])

    return MSASIBacktestResult(
        trades=trades,
        long_trades=long_trades,
        short_trades=short_trades,
        summary=summary,
        long_summary=long_summary,
        short_summary=short_summary,
        daily_pnl=daily_pnl,
        perf_stats=perf,
    )


def _compute_trade_summary(trades: List[TradeRecord], label: str) -> dict:
    """Compute summary statistics for a set of trades."""
    if not trades:
        return {"label": label, "n_trades": 0}

    returns = [t.strategy_return for t in trades]
    n = len(returns)
    wins = sum(1 for r in returns if r > 0)
    cumulative = 1.0
    for r in returns:
        cumulative *= (1.0 + r)
    cumulative_ret = cumulative - 1.0

    return {
        "label": label,
        "n_trades": n,
        "hit_rate": wins / n if n > 0 else 0.0,
        "cumulative_return": cumulative_ret,
        "avg_return": np.mean(returns),
        "median_return": np.median(returns),
        "max_return": max(returns),
        "min_return": min(returns),
        "avg_holding_days": np.mean([t.holding_days for t in trades]),
        "wins": wins,
        "losses": n - wins,
    }


def _build_daily_pnl(
    trades: List[TradeRecord],
    prices: pd.DataFrame,
    holding_period: int,
    trading_cost_bps: float,
    short_borrow_bps: float,
) -> pd.DataFrame:
    """Reconstruct daily P&L from trades for canonical perf stats.

    On days when a trade is active, the daily return is the CSI300 daily
    return multiplied by the trade direction (+1 for LONG, -1 for SHORT).
    Subtracts execution cost on entry date and daily short borrow cost.
    On days with no active trade, strategy return is 0.
    """
    if not trades:
        return pd.DataFrame(columns=["date", "strategy_ret"])

    prices = prices.sort_values("date").reset_index(drop=True)
    prices["daily_ret"] = prices["close"].pct_change()

    # Build a per-day direction map
    date_direction = {}
    trade_entries = set()
    for t in trades:
        trade_entries.add(t.entry_date)
        direction = 1.0 if t.signal_type == "LONG" else -1.0
        mask = (prices["date"] >= t.entry_date) & (prices["date"] <= t.exit_date)
        for d in prices.loc[mask, "date"]:
            # If overlapping trades, take the latest signal's direction
            date_direction[d] = direction

    daily_short_cost = (short_borrow_bps / 10000.0) / 252.0
    exec_cost = trading_cost_bps / 10000.0

    pnl_rows = []
    for _, row in prices.iterrows():
        d = row["date"]
        daily_r = row["daily_ret"] if np.isfinite(row["daily_ret"]) else 0.0
        direction = date_direction.get(d, 0.0)
        
        strat_r = daily_r * direction
        
        if direction != 0.0:
            if d in trade_entries:
                strat_r -= exec_cost
            if direction < 0.0:
                strat_r -= daily_short_cost
                
        pnl_rows.append({
            "date": d,
            "strategy_ret": strat_r,
            "benchmark_ret": daily_r,
            "in_trade": direction != 0.0,
        })

    return pd.DataFrame(pnl_rows)


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_backtest_report(result: MSASIBacktestResult) -> str:
    """Generate a human-readable backtest report."""
    lines = []
    lines.append("=" * 70)
    lines.append("MSASI 2.0 — Backtest Report")
    lines.append("=" * 70)

    for label, summary, trade_list in [
        ("ALL TRADES", result.summary, result.trades),
        ("LONG TRADES (Oversold)", result.long_summary, result.long_trades),
        ("SHORT TRADES (Overbought)", result.short_summary, result.short_trades),
    ]:
        lines.append(f"\n--- {label} ---")
        if summary.get("n_trades", 0) == 0:
            lines.append("  No trades.")
            continue

        lines.append(f"  Number of trades:    {summary['n_trades']}")
        lines.append(f"  Hit rate:            {summary['hit_rate']:.1%}")
        lines.append(f"  Cumulative return:   {summary['cumulative_return']:.1%}")
        lines.append(f"  Average return:      {summary['avg_return']:.2%}")
        lines.append(f"  Median return:       {summary['median_return']:.2%}")
        lines.append(f"  Best trade:          {summary['max_return']:.2%}")
        lines.append(f"  Worst trade:         {summary['min_return']:.2%}")
        lines.append(f"  Avg holding days:    {summary['avg_holding_days']:.0f}")
        lines.append(f"  Wins / Losses:       {summary['wins']} / {summary['losses']}")

        if trade_list:
            lines.append(f"\n  Trade details:")
            lines.append(f"  {'#':>3} {'Type':>5} {'Signal Date':>12} {'Entry':>12} {'Exit':>12} {'Return':>8} {'MSASI':>6}")
            lines.append(f"  {'-'*3} {'-'*5} {'-'*12} {'-'*12} {'-'*12} {'-'*8} {'-'*6}")
            for i, t in enumerate(trade_list, 1):
                lines.append(
                    f"  {i:3d} {t.signal_type:>5} {t.signal_date.strftime('%Y-%m-%d'):>12} "
                    f"{t.entry_date.strftime('%Y-%m-%d'):>12} {t.exit_date.strftime('%Y-%m-%d'):>12} "
                    f"{t.strategy_return:>7.2%} {t.msasi_at_signal:>5.1f}"
                )

    if result.perf_stats:
        lines.append(f"\n--- CANONICAL PERF STATS ---")
        lines.append(f"  CAGR:           {result.perf_stats.cagr:.2%}")
        lines.append(f"  Sharpe:         {result.perf_stats.sharpe:.3f}")
        lines.append(f"  Max Drawdown:   {result.perf_stats.max_drawdown:.2%}")
        lines.append(f"  Win Rate:       {result.perf_stats.win_rate:.2%}")

    report = "\n".join(lines)
    return report
