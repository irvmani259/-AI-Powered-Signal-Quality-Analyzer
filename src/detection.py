"""
detection.py
------------
Anomaly detection, interference pattern identification, and degradation
trend detection for signal quality data.

Detectors
---------
1. Rule-based thresholds (SNR, BER, latency, packet_loss)
2. Rolling z-score / EWMA control chart per node
3. Isolation Forest (global, trained on ingested data)
4. Interference pattern detector (periodic bursts, correlated SNR↓ + BER↑)
5. Degradation trend detector (sustained downward SNR / rising BER/latency)

Severity scoring
----------------
Each flagged row accumulates a score; the final severity label is:
  score 1   → Low
  score 2   → Medium
  score 3-4 → High
  score ≥5  → Critical
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (mirrors ingestion.THRESHOLDS for rule-based checks)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "snr_min": 10.0,
    "ber_max": 1e-3,
    "latency_max": 150.0,
    "packet_loss_max": 5.0,
}

# Z-score threshold for rolling anomaly
Z_THRESH = 3.0
# Isolation Forest contamination estimate
IF_CONTAMINATION = 0.05
# Minimum consecutive degraded samples to call a trend
DEGRADE_MIN_CONSECUTIVE = 6
# Minimum period regularity to call interference "periodic" (correlation threshold)
PERIODIC_CORR_THRESH = 0.35


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Container for all detection outputs."""
    anomalies: pd.DataFrame          # rows with anomaly flags + severity
    summary: list[dict[str, Any]]    # per-anomaly structured summary
    interference_events: list[dict[str, Any]]
    degradation_events: list[dict[str, Any]]
    node_health: pd.DataFrame        # per-node health score 0-100


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect(df: pd.DataFrame) -> DetectionResult:
    """
    Run all detection algorithms on a cleaned, enriched DataFrame.

    Parameters
    ----------
    df : Output of ingestion.ingest()

    Returns
    -------
    DetectionResult
    """
    df = df.copy()

    # 1. Rule-based flags
    df = _rule_based(df)

    # 2. Rolling z-score
    df = _rolling_zscore(df)

    # 3. Isolation Forest
    df = _isolation_forest(df)

    # 4. Combined severity score
    df = _score_severity(df)

    # 5. Interference patterns
    interference = _detect_interference(df)

    # 6. Degradation trends
    degradation = _detect_degradation(df)

    # 7. Node health
    node_health = _compute_node_health(df)

    # 8. Build anomaly table (rows with any flag)
    anomaly_cols = [c for c in df.columns if c.startswith("flag_") or c == "severity"]
    flag_mask = df[[c for c in df.columns if c.startswith("flag_")]].any(axis=1)
    anomalies = df[flag_mask].copy()

    summary = _build_summary(anomalies)

    logger.info(
        "Detection complete: %d anomalous rows / %d total | "
        "%d interference events | %d degradation events",
        len(anomalies), len(df), len(interference), len(degradation),
    )

    return DetectionResult(
        anomalies=anomalies,
        summary=summary,
        interference_events=interference,
        degradation_events=degradation,
        node_health=node_health,
    )


# ---------------------------------------------------------------------------
# 1. Rule-based thresholds
# ---------------------------------------------------------------------------

def _rule_based(df: pd.DataFrame) -> pd.DataFrame:
    df["flag_snr_low"] = df["snr"] < THRESHOLDS["snr_min"]
    df["flag_ber_high"] = df["ber"] > THRESHOLDS["ber_max"]
    df["flag_latency_high"] = df["latency"] > THRESHOLDS["latency_max"]
    if "packet_loss" in df.columns:
        df["flag_packet_loss_high"] = df["packet_loss"] > THRESHOLDS["packet_loss_max"]
    return df


# ---------------------------------------------------------------------------
# 2. Rolling z-score anomaly
# ---------------------------------------------------------------------------

def _rolling_zscore(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Per-node rolling z-score for SNR, BER, latency.
    A row is flagged if any metric's z-score exceeds Z_THRESH.
    """
    def _node_zscore(g: pd.DataFrame) -> pd.Series:
        # Work on the metric columns only (exclude group key to avoid pandas 2 warning)
        flags = pd.Series(False, index=g.index)
        for col in ["snr", "ber", "latency"]:
            roll_mean = g[col].rolling(window, min_periods=5).mean()
            roll_std = g[col].rolling(window, min_periods=5).std().replace(0, np.nan)
            z = ((g[col] - roll_mean) / roll_std).abs()
            flags |= z > Z_THRESH
        return flags

    result = (
        df.groupby("node_id", group_keys=False)
        .apply(_node_zscore, include_groups=False)
    )
    # pandas 2.2+: apply with include_groups=False may return a Series or DataFrame
    # squeeze to 1-D if needed
    if isinstance(result, pd.DataFrame):
        result = result.iloc[:, 0]
    df["flag_zscore"] = result.reindex(df.index).infer_objects(copy=False).fillna(False).astype(bool)
    return df


# ---------------------------------------------------------------------------
# 3. Isolation Forest
# ---------------------------------------------------------------------------

def _isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train an Isolation Forest on (snr, ber_log, latency) and flag inliers/outliers.
    Uses log-transform on BER since it spans many orders of magnitude.
    """
    feature_cols = ["snr", "latency"]
    X = df[feature_cols].copy()
    X["ber_log"] = np.log10(df["ber"].clip(lower=1e-12))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = IsolationForest(
        contamination=IF_CONTAMINATION,
        n_estimators=100,
        random_state=42,
    )
    preds = clf.fit_predict(Xs)
    scores = clf.decision_function(Xs)  # more negative = more anomalous

    df["flag_isolation"] = preds == -1
    df["if_score"] = scores  # keep for severity weighting
    return df


# ---------------------------------------------------------------------------
# 4. Severity scoring
# ---------------------------------------------------------------------------

def _score_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Accumulate boolean flags into a numeric score then assign a label."""
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    df["severity_score"] = df[flag_cols].sum(axis=1)

    # Boost score for extreme values
    df.loc[df["snr"] < 5, "severity_score"] += 1
    df.loc[df["ber"] > 1e-2, "severity_score"] += 1
    df.loc[df["latency"] > 300, "severity_score"] += 1

    def _label(s: int) -> str:
        if s == 0:
            return "Normal"
        if s == 1:
            return "Low"
        if s == 2:
            return "Medium"
        if s <= 4:
            return "High"
        return "Critical"

    df["severity"] = df["severity_score"].apply(_label)
    return df


# ---------------------------------------------------------------------------
# 5. Interference pattern detection
# ---------------------------------------------------------------------------

def _detect_interference(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Detect two interference signatures per node:
    a) Correlated SNR drop + BER spike (simultaneous)
    b) Periodic/bursty pattern via autocorrelation of the anomaly signal
    """
    events: list[dict[str, Any]] = []

    for node, g in df.groupby("node_id"):
        g = g.sort_values("timestamp")
        # --- a) Correlated SNR↓ + BER↑ ---
        snr_low = (g["snr"] < THRESHOLDS["snr_min"]).astype(int)
        ber_high = (g["ber"] > THRESHOLDS["ber_max"]).astype(int)
        # Guard: corr is undefined when one series has zero variance
        if snr_low.std() == 0 or ber_high.std() == 0:
            corr = 0.0
        else:
            corr = float(snr_low.corr(ber_high))
        if corr > 0.4:
            events.append({
                "type": "correlated_snr_ber",
                "node_id": node,
                "correlation": round(float(corr), 3),
                "description": (
                    f"Node {node}: SNR drops strongly correlate with BER spikes "
                    f"(r={corr:.2f}) — likely co-channel interference or hardware issue."
                ),
            })

        # --- b) Periodic pattern via autocorrelation ---
        anomaly_signal = (
            g["flag_snr_low"].astype(int) | g["flag_ber_high"].astype(int)
        ).to_numpy()
        if len(anomaly_signal) > 40 and anomaly_signal.sum() > 5:
            # Compute autocorrelation up to lag 60
            n = len(anomaly_signal)
            centered = anomaly_signal - anomaly_signal.mean()
            ac = np.correlate(centered, centered, mode="full")
            ac = ac[n - 1:]
            ac /= ac[0] + 1e-12
            # Look for a secondary peak beyond lag 5
            if len(ac) > 60:
                peak_lag = int(np.argmax(ac[5:61]) + 5)
                peak_val = float(ac[peak_lag])
                if peak_val > PERIODIC_CORR_THRESH:
                    events.append({
                        "type": "periodic_interference",
                        "node_id": node,
                        "period_samples": peak_lag,
                        "autocorr_peak": round(peak_val, 3),
                        "description": (
                            f"Node {node}: Periodic anomaly pattern detected every "
                            f"~{peak_lag} samples (autocorr={peak_val:.2f}) — "
                            "possible scheduled task, hardware cycle, or co-channel source."
                        ),
                    })

    return events


# ---------------------------------------------------------------------------
# 6. Degradation trend detection
# ---------------------------------------------------------------------------

def _detect_degradation(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Flag sustained downward SNR trends and rising BER/latency trends
    using the pre-computed *_trend columns from ingestion.
    """
    events: list[dict[str, Any]] = []

    for node, g in df.groupby("node_id"):
        g = g.sort_values("timestamp")

        # SNR trend: negative slope means SNR is falling
        snr_degrading = g["snr_trend"] < -0.05
        run = _max_consecutive(snr_degrading)
        if run >= DEGRADE_MIN_CONSECUTIVE:
            worst = g.loc[snr_degrading, "snr"].min()
            events.append({
                "type": "snr_degradation",
                "node_id": node,
                "max_consecutive_samples": int(run),
                "min_snr_observed": round(float(worst), 2),
                "description": (
                    f"Node {node}: Sustained SNR decline detected over {run} consecutive "
                    f"samples. Minimum SNR = {worst:.1f} dB."
                ),
            })

        # BER trend: positive slope means BER is rising
        ber_degrading = g["ber_trend"] > 1e-7
        run = _max_consecutive(ber_degrading)
        if run >= DEGRADE_MIN_CONSECUTIVE:
            worst = g.loc[ber_degrading, "ber"].max()
            events.append({
                "type": "ber_degradation",
                "node_id": node,
                "max_consecutive_samples": int(run),
                "max_ber_observed": float(worst),
                "description": (
                    f"Node {node}: Sustained BER increase over {run} consecutive samples. "
                    f"Peak BER = {worst:.2e}."
                ),
            })

        # Latency trend
        lat_degrading = g["latency_trend"] > 0.5
        run = _max_consecutive(lat_degrading)
        if run >= DEGRADE_MIN_CONSECUTIVE:
            worst = g.loc[lat_degrading, "latency"].max()
            events.append({
                "type": "latency_degradation",
                "node_id": node,
                "max_consecutive_samples": int(run),
                "max_latency_observed": round(float(worst), 2),
                "description": (
                    f"Node {node}: Latency trending upward over {run} samples. "
                    f"Peak latency = {worst:.0f} ms."
                ),
            })

    return events


def _max_consecutive(mask: pd.Series) -> int:
    """Return the length of the longest consecutive True run in a boolean Series."""
    max_run = cur_run = 0
    for v in mask:
        if v:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


# ---------------------------------------------------------------------------
# 7. Node health score
# ---------------------------------------------------------------------------

def _compute_node_health(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a 0-100 health score per node.
    Score = 100 minus weighted penalty for anomaly rate and severity distribution.
    """
    records = []
    flag_cols = [c for c in df.columns if c.startswith("flag_")]

    for node, g in df.groupby("node_id"):
        total = len(g)
        if total == 0:
            continue

        anomaly_rate = g[flag_cols].any(axis=1).mean()
        critical_rate = (g["severity"] == "Critical").mean()
        high_rate = (g["severity"] == "High").mean()

        # Penalty weights
        penalty = (
            anomaly_rate * 40
            + critical_rate * 35
            + high_rate * 15
            + (g["snr"] < THRESHOLDS["snr_min"]).mean() * 20
            + (g["ber"] > THRESHOLDS["ber_max"]).mean() * 20
        )
        health = max(0.0, min(100.0, 100.0 - penalty * 100))

        records.append({
            "node_id": node,
            "health_score": round(health, 1),
            "anomaly_rate": round(anomaly_rate * 100, 1),
            "critical_pct": round(critical_rate * 100, 1),
            "high_pct": round(high_rate * 100, 1),
            "mean_snr": round(g["snr"].mean(), 2),
            "mean_ber": float(g["ber"].mean()),
            "mean_latency": round(g["latency"].mean(), 2),
            "total_samples": total,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 8. Summary builder
# ---------------------------------------------------------------------------

def _build_summary(anomalies: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert flagged rows to a list of structured dicts for the API response."""
    if anomalies.empty:
        return []

    flag_cols = [c for c in anomalies.columns if c.startswith("flag_")]
    records = []
    for _, row in anomalies.iterrows():
        active_flags = [c.replace("flag_", "") for c in flag_cols if row.get(c, False)]
        records.append({
            "timestamp": row["timestamp"].isoformat(),
            "node_id": row["node_id"],
            "snr": round(float(row["snr"]), 3),
            "ber": float(row["ber"]),
            "latency": round(float(row["latency"]), 2),
            "severity": row["severity"],
            "flags": active_flags,
        })
    return records
