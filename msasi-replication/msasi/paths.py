"""Local data paths.

Small stand-in for the path helper the original project used. Only the four
functions this package calls are kept, and they resolve to the same sub-path
layout the readers were written against.

Root directory: environment variable ``MSASI_DATA_ROOT``; default ``./data``
(relative to the current working directory).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Union

_StrPath = Union[str, Path]

# (market, category) -> base segments under the data root
_ROUTING: dict[tuple[str, str], tuple[str, ...]] = {
    ("cn", "prices"):          ("cn", "equities", "prices"),
    ("cn", "index_prices"):    ("cn", "indices", "prices"),
    ("cn", "northbound"):      ("cn", "flows", "northbound"),
    ("cn", "financing"):       ("cn", "flows", "financing"),
    ("cn", "sse_new_account"): ("cn", "flows", "sse_new_account"),
    ("cn", "index_futures"):   ("cn", "derivatives", "index_futures"),
    ("cn", "index_options"):   ("cn", "derivatives", "index_options"),
}


@lru_cache(maxsize=None)
def data_root() -> Path:
    """Return the absolute data root (``MSASI_DATA_ROOT`` or ``./data``)."""
    p = Path(os.environ.get("MSASI_DATA_ROOT", "").strip() or "data")
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def _build(base: Path, parts: Iterable[_StrPath]) -> Path:
    """Join parts to base, rejecting absolute segments."""
    segs = []
    for part in parts:
        s = part.as_posix() if isinstance(part, Path) else str(part)
        if Path(s).is_absolute():
            raise ValueError(
                f"Absolute path segment '{s}' not allowed; supply relative components only."
            )
        segs.append(s)
    return (base.joinpath(*segs)).resolve() if segs else base.resolve()


def data_path(country: str, *parts: _StrPath) -> Path:
    """Return ``data_root / country / *parts``.

    e.g. ``data_path("cn", "equities", "prices", "raw")`` → ``<root>/cn/equities/prices/raw``
    """
    return _build(data_root() / country, parts)


def cache_path(market: str, *parts: _StrPath) -> Path:
    """Raw-stage path: ``<root>/<routing for (market, category)>/raw/...``."""
    if not parts:
        raise ValueError("cache_path requires at least a category argument")
    category = str(parts[0])
    base_segs = _ROUTING.get((market.lower(), category.lower()), (market, "equities", category))
    return _build(data_root() / Path(*base_segs) / "raw", parts[1:])


def silver_path(market: str, *parts: _StrPath) -> Path:
    """Clean-stage path: ``<root>/<routing for (market, dataset)>/clean/...``.

    e.g. ``silver_path("cn", "northbound")`` → ``<root>/cn/flows/northbound/clean``
    """
    if not parts:
        raise ValueError("silver_path requires at least a dataset argument")
    dataset = str(parts[0])
    base_segs = _ROUTING.get((market.lower(), dataset.lower()), (market, "equities", dataset))
    return _build(data_root() / Path(*base_segs) / "clean", parts[1:])
