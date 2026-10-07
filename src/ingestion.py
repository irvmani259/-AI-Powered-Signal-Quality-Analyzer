"""
ingestion.py
------------
Handles CSV/JSON upload validation, cleaning, normalization,
rolling statistics computation, and synthetic data generation.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = {"timestamp", "node_id", "snr", "ber", "latency"}
OPTIONAL_COLUMNS = {"frequency_band", "location", "packet_loss"}

COLUMN_ALIASES: dict[str, str] = {
    "snr_db": "snr",
    "snr(db)": "snr",
    "latency_ms": "latency",
    "latency(ms)": "latency",
}

# Rule-based thresholds (used here for outlier context and downstream)
THRESHOLDS = {
    "snr_min": 10.0,       # dB — below this is degraded
    "ber_max": 1e-3,       # above this is degraded
    "latency_max": 150.0,  # ms — above this is degraded
    "packet_loss_max": 5.0,# % — above this is degraded
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest(source: str | bytes | pd.DataFrame, fmt: str = "csv") -> pd.DataFrame:
    """
    Load, validate, clean, normalize, and enrich signal data.

    Parameters
    ----------
    source : path string, raw bytes, or an existing DataFrame
    fmt    : 'csv' or 'json' (ignored when source is a DataFrame)

    Returns
    -------
    Cleaned and enriched DataFrame ready for detection / forecasting.
    """
    df = _load(source, fmt)
    df = _normalize_columns(df)
    df = _validate_schema(df)
    df = _parse_timestamps(df)
    df = _impute_and_clip(df)
    df = _remove_outliers(df)
    df = _sort(df)
    df = _add_rolling_stats(df)
    logger.info("Ingestion complete: %d rows, %d nodes", len(df), df["node_id"].nunique())
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load(source: str | bytes | pd.DataFrame, fmt: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (bytes, bytearray)):
        buf = io.BytesIO(source)
        return pd.read_json(buf) if fmt == "json" else pd.read_csv(buf)
    # file path
    return pd.read_json(source) if fmt == "json" else pd.read_csv(source)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case column names and apply known aliases."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns=COLUMN_ALIASES)
    return df


def _validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    bad = df["timestamp"].isna().sum()
    if bad > 0:
        logger.warning("Dropping %d rows with unparseable timestamps", bad)
        df = df.dropna(subset=["timestamp"])
    return df


def _impute_and_clip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing numeric values with per-node forward-fill then backward-fill,
    then fall back to column median.  Clip values to physically plausible ranges.
    """
    numeric = ["snr", "ber", "latency"]
    optional_numeric = [c for c in ["packet_loss"] if c in df.columns]
    all_numeric = numeric + optional_numeric

    # Per-node ffill/bfill
    df = df.sort_values(["node_id", "timestamp"])
    df[all_numeric] = (
        df.groupby("node_id")[all_numeric]
        .transform(lambda g: g.ffill().bfill())
    )
    # Global median fallback
    for col in all_numeric:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # Physical clipping
    df["snr"] = df["snr"].clip(lower=-10, upper=60)
    df["ber"] = df["ber"].clip(lower=0, upper=1)
    df["latency"] = df["latency"].clip(lower=0, upper=5000)
    if "packet_loss" in df.columns:
        df["packet_loss"] = df["packet_loss"].clip(lower=0, upper=100)

    return df


def _remove_outliers(df: pd.DataFrame, z_thresh: float = 5.0) -> pd.DataFrame:
    """Flag (but keep) extreme statistical outliers via z-score."""
    for col in ["snr", "ber", "latency"]:
        z = np.abs(stats.zscore(df[col].fillna(df[col].median())))
        df[f"{col}_outlier"] = z > z_thresh
    return df


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)


def _add_rolling_stats(df: pd.DataFrame, windows: list[int] = [5, 15, 30]) -> pd.DataFrame:
    """
    Per node, compute rolling mean, std, and EWMA trend for SNR, BER, latency.
    Also compute a simple linear trend coefficient over the last 30 samples.
    """
    metrics = ["snr", "ber", "latency"]
    df = df.sort_values(["node_id", "timestamp"])

    for w in windows:
        for m in metrics:
            grp = df.groupby("node_id")[m]
            df[f"{m}_roll_mean_{w}"] = grp.transform(
                lambda g, _w=w: g.rolling(_w, min_periods=1).mean()
            )
            df[f"{m}_roll_std_{w}"] = grp.transform(
                lambda g, _w=w: g.rolling(_w, min_periods=1).std().fillna(0)
            )
            df[f"{m}_ewma"] = grp.transform(
                lambda g: g.ewm(span=10, adjust=False).mean()
            )

    # Linear trend over last 30 samples per node
    def _trend_slope(series: pd.Series) -> pd.Series:
        slopes = pd.Series(index=series.index, dtype=float)
        arr = series.to_numpy()
        for i in range(len(arr)):
            start = max(0, i - 29)
            segment = arr[start: i + 1]
            if len(segment) >= 3:
                x = np.arange(len(segment))
                slope, *_ = np.polyfit(x, segment, 1)
            else:
                slope = 0.0
            slopes.iloc[i] = slope
        return slopes

    for m in metrics:
        df[f"{m}_trend"] = df.groupby("node_id")[m].transform(_trend_slope)

    return df


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_synthetic(
    n_nodes: int = 5,
    hours: int = 48,
    freq_minutes: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic signal-quality dataset.

    Scenario mix per node (randomly assigned):
        - normal          : stable signal with mild noise
        - gradual_degrade : SNR slowly drops, BER / latency slowly rises
        - sudden_drop     : one or more abrupt step changes
        - interference    : periodic bursts of noise
        - bursty_noise    : random high-frequency spikes
    """
    rng = np.random.default_rng(seed)
    scenarios = ["normal", "gradual_degrade", "sudden_drop", "interference", "bursty_noise"]
    node_scenarios = {f"NODE_{i+1:02d}": scenarios[i % len(scenarios)] for i in range(n_nodes)}

    freq = pd.Timedelta(minutes=freq_minutes)
    n_points = int(hours * 60 / freq_minutes)
    timestamps = pd.date_range("2024-01-01", periods=n_points, freq=freq, tz="UTC")

    rows = []
    for node, scenario in node_scenarios.items():
        snr_base, ber_base, lat_base = 28.0, 1e-5, 40.0
        snr = np.full(n_points, snr_base)
        ber = np.full(n_points, ber_base)
        lat = np.full(n_points, lat_base)

        if scenario == "normal":
            snr += rng.normal(0, 1.5, n_points)
            ber *= np.exp(rng.normal(0, 0.3, n_points))
            lat += rng.normal(0, 5, n_points)

        elif scenario == "gradual_degrade":
            slope = rng.uniform(0.08, 0.15)
            snr -= slope * np.arange(n_points) + rng.normal(0, 1.2, n_points)
            ber *= np.exp(slope * 0.05 * np.arange(n_points)) * np.exp(rng.normal(0, 0.3, n_points))
            lat += slope * 0.8 * np.arange(n_points) + rng.normal(0, 4, n_points)

        elif scenario == "sudden_drop":
            snr += rng.normal(0, 1.2, n_points)
            drop_times = rng.integers(n_points // 4, 3 * n_points // 4, size=2)
            for dt in drop_times:
                snr[dt:dt + 20] -= rng.uniform(10, 18)
                ber[dt:dt + 20] *= rng.uniform(50, 200)
                lat[dt:dt + 20] += rng.uniform(80, 200)

        elif scenario == "interference":
            # Periodic 30-min bursts
            period = int(30 / freq_minutes)
            for i in range(0, n_points, period):
                burst_len = int(period * 0.25)
                snr[i:i + burst_len] -= rng.uniform(5, 12)
                ber[i:i + burst_len] *= rng.uniform(10, 50)
                lat[i:i + burst_len] += rng.uniform(30, 80)
            snr += rng.normal(0, 1.0, n_points)

        elif scenario == "bursty_noise":
            spike_idx = rng.choice(n_points, size=int(n_points * 0.08), replace=False)
            snr[spike_idx] -= rng.uniform(8, 20, len(spike_idx))
            ber[spike_idx] *= rng.uniform(20, 100, len(spike_idx))
            lat[spike_idx] += rng.uniform(60, 180, len(spike_idx))
            snr += rng.normal(0, 1.3, n_points)

        snr = np.clip(snr, -5, 55)
        ber = np.clip(ber, 1e-9, 1.0)
        lat = np.clip(lat, 5, 3000)

        for i, ts in enumerate(timestamps):
            rows.append({
                "timestamp": ts,
                "node_id": node,
                "snr": round(float(snr[i]), 3),
                "ber": float(ber[i]),
                "latency": round(float(lat[i]), 2),
                "frequency_band": rng.choice(["700MHz", "1800MHz", "2600MHz", "3500MHz"]),
                "location": rng.choice(["SectorA", "SectorB", "SectorC"]),
                "packet_loss": round(float(np.clip(rng.normal(0.5, 0.3), 0, 10)), 2),
                "scenario": scenario,
            })

    df = pd.DataFrame(rows)
    logger.info("Generated synthetic dataset: %d rows, %d nodes", len(df), n_nodes)
    return df
