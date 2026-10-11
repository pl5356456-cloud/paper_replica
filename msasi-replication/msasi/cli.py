"""MSASI 2.0 — CLI runner (production).

Usage::

    # Daily signal generation (cron)
    python -m msasi daily
    python -m msasi daily --date 2026-04-13

    # Full pipeline + backtest (research mode)
    python -m msasi run --backtest

    # Pipeline only (no backtest)
    python -m msasi run

    # Custom date range
    python -m msasi run --start 2014-01-01 --end 2026-03-31 --backtest
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party logs
    logging.getLogger("pyarrow").setLevel(logging.WARNING)


# ── Subcommands ──────────────────────────────────────────────────


def cmd_daily(args) -> None:
    """Daily signal generation (production cron entry point)."""
    from .signal import generate_daily_signal
    from .alerts import (
        check_sentiment_extreme,
        format_sentiment_alert,
        save_alert_log,
    )

    snapshot = generate_daily_signal(asof_date=args.date, save=True)

    print(f"\n{'=' * 60}")
    print("MSASI 2.0 — Daily Signal")
    print(f"{'=' * 60}")
    print(f"  Date:       {snapshot.date.date()}")
    print(f"  MSASI raw:  {snapshot.msasi_raw:.1f}%")
    print(f"  MSASI 1MMA: {snapshot.msasi_1mma:.1f}%")
    print(f"  Zone:       {snapshot.zone}")
    print(f"  Signal:     {snapshot.active_signal or '—'}")

    # Check for sentiment extreme
    alert = check_sentiment_extreme(
        current_date=snapshot.date.date(),
        current_msasi_1mma=snapshot.msasi_1mma,
        current_zone=snapshot.zone,
    )
    if alert:
        msg = format_sentiment_alert(alert)
        print(msg)
        save_alert_log(alert)


def cmd_run(args) -> None:
    """Full pipeline + optional backtest (research mode)."""
    from .data.readers import load_all_indicators, load_allshare_prices
    from .engine import MSASIEngine
    from .backtest import run_msasi_backtest, print_backtest_report

    # Clear cache if requested
    if args.clear_cache:
        import shutil
        from .data.readers import _msasi_cache_dir
        cache_dir = _msasi_cache_dir()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logging.info("Cleared MSASI aggregate cache at %s", cache_dir)

    # Load data
    logging.info("Loading all indicators (start=%s, end=%s)...", args.start, args.end)
    raw = load_all_indicators(start_date=args.start, end_date=args.end)
    allshare = load_allshare_prices()

    if raw.empty:
        logging.error("No indicator data loaded. Aborting.")
        sys.exit(1)

    # Run engine
    use_expanding = not args.static_weights
    engine = MSASIEngine()
    result = engine.run(raw, allshare, use_expanding_weights=use_expanding)

    # Print summary
    print(f"\n{'=' * 70}")
    print("MSASI 2.0 — Summary")
    print(f"{'=' * 70}")
    print(f"Date range:     {result.msasi_daily['date'].min().date()} → "
          f"{result.msasi_daily['date'].max().date()}")
    print(f"Trading days:   {len(result.msasi_daily)}")
    print(f"Current MSASI:  raw={result.msasi_daily['msasi_raw'].iloc[-1]:.1f}%, "
          f"1MMA={result.msasi_daily['msasi_1mma'].iloc[-1]:.1f}%")
    print(f"Total signals:  {len(result.signals)} "
          f"({(result.signals['signal_type'] == 'LONG').sum()} LONG, "
          f"{(result.signals['signal_type'] == 'SHORT').sum()} SHORT)")

    print("\nFinal indicator weights:")
    for key, weight in result.weights.sort_values(ascending=False).items():
        print(f"  {key:30s}  {weight:.4f}")

    if not result.signals.empty:
        print("\nSignal dates:")
        for _, sig in result.signals.iterrows():
            print(f"  {sig['date'].strftime('%Y-%m-%d')}  {sig['signal_type']:>5}  "
                  f"MSASI 1MMA={sig['msasi_1mma']:.1f}%")

    # Backtest
    if args.backtest:
        print(f"\n{'=' * 70}")
        print("Running backtest...")
        bt_result = run_msasi_backtest(result.signals, allshare)
        report = print_backtest_report(bt_result)
        print(report)

    # Output logic
    if args.output:
        from pathlib import Path
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        daily_path = out_dir / "msasi_daily.parquet"
        result.msasi_daily.to_parquet(daily_path, index=False)
        logging.info("Saved daily indicators to %s", daily_path)

        signals_path = out_dir / "msasi_signals.parquet"
        result.signals.to_parquet(signals_path, index=False)
        logging.info("Saved signal records to %s", signals_path)

    print("\nDone.")


# ── Main entry point ─────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m msasi",
        description="MSASI 2.0 — CN Market Sentiment Indicator (Production)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # daily (production)
    p = sub.add_parser("daily", help="Generate daily MSASI signal (cron)")
    p.add_argument("--date", type=str, default=None,
                    help="As-of date (YYYY-MM-DD). Default: latest available.")
    p.set_defaults(func=cmd_daily)

    # run (research)
    p = sub.add_parser("run", help="Full pipeline + optional backtest (research)")
    p.add_argument("--start", type=str, default="2013-01-01",
                    help="Start date (default: 2013-01-01, allows norm window warmup)")
    p.add_argument("--end", type=str, default=None,
                    help="End date (default: latest available)")
    p.add_argument("--backtest", action="store_true",
                    help="Run event-driven backtest after computing MSASI")
    p.add_argument("--expanding-weights", action="store_true", default=True,
                    help="Use expanding-window weights (default: True)")
    p.add_argument("--static-weights", action="store_true",
                    help="Use static (full-sample) weights instead of expanding")
    p.add_argument("--clear-cache", action="store_true",
                    help="Clear pre-computed aggregate cache before running")
    # Resolve default output via the local paths module
    from .paths import data_root, data_path
    _default_output = str(data_path("cn", "equities", "sentiment", "msasi"))
    p.add_argument("--output", type=str, default=_default_output,
                    help="Directory to save output parquet files")
    p.set_defaults(func=cmd_run)

    args = parser.parse_args()
    _setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
