"""MSASI 2.0 — CN Market Sentiment Indicator.

Replicates Morgan Stanley's A-Share Sentiment Indicator (MSASI 2.0)
using 11 component indicators with 100-day rolling min-max normalization,
R²-weighted aggregation, and extreme-signal generation.

Methodology (from paper):
    1. Normalize all indicators via 100-day rolling min-max → [0, 100]
    2. Weight by expanding-window R² vs CSI300 forward returns
    3. Aggregate → MSASI raw + MSASI 1MMA (20-day moving average)
    4. Signal: MSASI 1MMA ≤ 25% → LONG; ≥ 75% → SHORT (70-day hold)

Usage (research)::

    from msasi.engine import MSASIEngine, MSASIResult
    from msasi.data.readers import load_all_indicators, load_allshare_prices
    from msasi.backtest import run_msasi_backtest

    raw = load_all_indicators(start_date="2013-01-01")
    csi = load_allshare_prices()
    engine = MSASIEngine()
    result = engine.run(raw, csi)
    bt = run_msasi_backtest(result.signals, csi)

Usage (production — daily cron)::

    from msasi.signal import generate_daily_signal
    snapshot = generate_daily_signal("2026-04-13")
"""
