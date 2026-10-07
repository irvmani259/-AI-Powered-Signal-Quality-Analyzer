"""
forecasting.py
--------------
Forecasts SNR, BER, and latency per node for the next N hours.

Strategy
--------
1. Try Prophet first (handles seasonality, holidays, missing data well).
2. Fall back to ARIMA (statsmodels) if Prophet is unavailable or fails.
3. Fall back to a simple gradient-boosting model with lag features if both fail.

Outputs
-------
- Per-node, per-metric forecast DataFrames with yhat, yhat_lower, yhat_upper
- Failure probability estimate (logistic model on forecast trajectory)
- Time-to-failure estimate (hours until first threshold breach)
- Risk ranking across all nodes
- Forecast accuracy metrics (MAE, RMSE) evaluated on a 20% holdout
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Suppress noisy convergence warnings from statsmodels/Prophet
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METRICS = ["snr", "ber", "latency"]
THRESHOLDS = {
    "snr": 10.0,      # failure if SNR drops below
    "ber": 1e-3,      # failure if BER rises above
    "latency": 150.0, # failure if latency rises above
}
FORECAST_HORIZON_HOURS = 12
MIN_TRAIN_SAMPLES = 30  # minimum points needed per metric


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    forecasts: dict[str, dict[str, pd.DataFrame]]  # node → metric → DataFrame
    accuracy: dict[str, dict[str, dict[str, float]]]  # node → metric → {mae, rmse}
    failure_probability: dict[str, float]              # node → probability 0-1
    time_to_failure: dict[str, float | None]           # node → hours (None if no breach)
    risk_ranking: pd.DataFrame                         # sorted risk table


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def forecast(
    df: pd.DataFrame,
    horizon_hours: int = FORECAST_HORIZON_HOURS,
) -> ForecastResult:
    """
    Generate forecasts and failure risk estimates for all nodes in df.

    Parameters
    ----------
    df           : Output of ingestion.ingest()
    horizon_hours: How many hours ahead to forecast

    Returns
    -------
    ForecastResult
    """
    all_forecasts: dict[str, dict[str, pd.DataFrame]] = {}
    all_accuracy: dict[str, dict[str, dict[str, float]]] = {}
    all_failure_prob: dict[str, float] = {}
    all_ttf: dict[str, float | None] = {}

    # Determine sampling interval in minutes
    df_sorted = df.sort_values("timestamp")
    timestamps = df_sorted["timestamp"].drop_duplicates().sort_values()
    if len(timestamps) > 1:
        diffs = timestamps.diff().dropna()
        freq_minutes = diffs.median().total_seconds() / 60
    else:
        freq_minutes = 5.0

    steps = int(horizon_hours * 60 / freq_minutes)

    for node, g in df.groupby("node_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        node_forecasts: dict[str, pd.DataFrame] = {}
        node_accuracy: dict[str, dict[str, float]] = {}

        for metric in METRICS:
            series = g[["timestamp", metric]].rename(
                columns={"timestamp": "ds", metric: "y"}
            )
            series = series.dropna(subset=["y"])

            if len(series) < MIN_TRAIN_SAMPLES:
                logger.warning("Node %s / %s: too few samples (%d), skipping", node, metric, len(series))
                continue

            # Holdout last 20% for accuracy evaluation
            holdout_n = max(1, int(len(series) * 0.2))
            train = series.iloc[:-holdout_n]
            holdout = series.iloc[-holdout_n:]

            fcast_df, acc = _forecast_single(train, holdout, steps, freq_minutes, metric)
            node_forecasts[metric] = fcast_df
            node_accuracy[metric] = acc

        all_forecasts[node] = node_forecasts
        all_accuracy[node] = node_accuracy

        # Failure probability & TTF
        prob, ttf = _estimate_failure_risk(node_forecasts, freq_minutes)
        all_failure_prob[node] = prob
        all_ttf[node] = ttf

    risk_df = _build_risk_ranking(all_failure_prob, all_ttf, df)

    logger.info(
        "Forecasting complete: %d nodes, horizon=%dh",
        len(all_forecasts), horizon_hours,
    )
    return ForecastResult(
        forecasts=all_forecasts,
        accuracy=all_accuracy,
        failure_probability=all_failure_prob,
        time_to_failure=all_ttf,
        risk_ranking=risk_df,
    )


# ---------------------------------------------------------------------------
# Single-metric forecaster (tries Prophet → ARIMA → GBM)
# ---------------------------------------------------------------------------

def _forecast_single(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    steps: int,
    freq_minutes: float,
    metric: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Try three forecasting strategies in order.
    Returns (forecast_df with columns [ds, yhat, yhat_lower, yhat_upper], accuracy_dict).
    """
    strategies = [_try_prophet, _try_arima, _try_gbm]
    last_err: Exception | None = None

    for strategy in strategies:
        try:
            fcast_df, acc = strategy(train, holdout, steps, freq_minutes, metric)
            return fcast_df, acc
        except Exception as exc:  # noqa: BLE001
            logger.debug("Strategy %s failed for metric %s: %s", strategy.__name__, metric, exc)
            last_err = exc

    # Ultimate fallback: flat line at last value
    logger.warning("All strategies failed for metric %s, using flat fallback: %s", metric, last_err)
    return _flat_fallback(train, holdout, steps, freq_minutes)


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

def _try_prophet(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    steps: int,
    freq_minutes: float,
    metric: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    from prophet import Prophet  # type: ignore

    log_transform = metric == "ber"
    t = train.copy()
    if log_transform:
        t["y"] = np.log10(t["y"].clip(lower=1e-12))

    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=True,
        interval_width=0.95,
        changepoint_prior_scale=0.05,
    )
    m.fit(t)

    freq_str = f"{int(freq_minutes)}min" if freq_minutes >= 1 else "30s"
    future = m.make_future_dataframe(periods=steps, freq=freq_str)
    forecast_raw = m.predict(future)

    fcast_df = forecast_raw[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(steps).copy()

    if log_transform:
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            fcast_df[col] = 10 ** fcast_df[col]

    # Accuracy on holdout
    hd = holdout.copy()
    if log_transform:
        hd["y"] = np.log10(hd["y"].clip(lower=1e-12))
    holdout_pred = m.predict(hd.rename(columns={"y": "y_true"})[["ds"]])
    y_pred = holdout_pred["yhat"].values
    y_true = (
        np.log10(holdout["y"].clip(lower=1e-12).values) if log_transform
        else holdout["y"].values
    )
    acc = _calc_accuracy(y_true, y_pred)
    return fcast_df, acc


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

def _try_arima(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    steps: int,
    freq_minutes: float,
    metric: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore

    log_transform = metric == "ber"
    y_train = train["y"].values
    if log_transform:
        y_train = np.log10(np.clip(y_train, 1e-12, None))

    model = ARIMA(y_train, order=(2, 1, 2))
    res = model.fit()

    fc = res.get_forecast(steps=steps + len(holdout))
    pred_mean = fc.predicted_mean
    conf_int = fc.conf_int(alpha=0.05)

    # Accuracy on holdout
    y_hold_pred = pred_mean[: len(holdout)]
    y_hold_true = (
        np.log10(np.clip(holdout["y"].values, 1e-12, None))
        if log_transform else holdout["y"].values
    )
    acc = _calc_accuracy(y_hold_true, y_hold_pred)

    # Future forecast
    y_future = pred_mean[len(holdout):]
    lo = conf_int.iloc[len(holdout):, 0].values
    hi = conf_int.iloc[len(holdout):, 1].values

    last_ts = train["ds"].max()
    freq = pd.Timedelta(minutes=freq_minutes)
    future_ds = [last_ts + freq * (i + 1) for i in range(len(y_future))]

    if log_transform:
        y_future = 10 ** y_future
        lo = 10 ** lo
        hi = 10 ** hi

    fcast_df = pd.DataFrame({"ds": future_ds, "yhat": y_future, "yhat_lower": lo, "yhat_upper": hi})
    return fcast_df, acc


# ---------------------------------------------------------------------------
# Gradient Boosting with lag features
# ---------------------------------------------------------------------------

def _try_gbm(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    steps: int,
    freq_minutes: float,
    metric: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    log_transform = metric == "ber"
    y_all = np.concatenate([train["y"].values, holdout["y"].values])
    if log_transform:
        y_all = np.log10(np.clip(y_all, 1e-12, None))

    lags = [1, 2, 3, 5, 10, 15]
    X, y = _make_lag_features(y_all[: len(train)], lags)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gbm = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    gbm.fit(X_scaled, y)

    # Holdout accuracy
    X_hold, y_hold_true = _make_lag_features(
        y_all[: len(train) + len(holdout)], lags, start=len(train)
    )
    y_hold_pred = gbm.predict(scaler.transform(X_hold))
    acc = _calc_accuracy(y_hold_true, y_hold_pred)

    # Recursive future forecast
    history = list(y_all[: len(train) + len(holdout)])
    preds, lowers, uppers = [], [], []
    residual_std = float(np.std(y_hold_true - y_hold_pred)) if len(y_hold_true) > 0 else 0.1

    for _ in range(steps):
        feat = np.array([[history[-l] for l in lags]])
        p = float(gbm.predict(scaler.transform(feat))[0])
        preds.append(p)
        lowers.append(p - 1.96 * residual_std)
        uppers.append(p + 1.96 * residual_std)
        history.append(p)

    if log_transform:
        preds = list(10 ** np.array(preds))
        lowers = list(10 ** np.array(lowers))
        uppers = list(10 ** np.array(uppers))

    last_ts = train["ds"].max()
    freq = pd.Timedelta(minutes=freq_minutes)
    future_ds = [last_ts + freq * (i + 1) for i in range(steps)]
    fcast_df = pd.DataFrame({"ds": future_ds, "yhat": preds, "yhat_lower": lowers, "yhat_upper": uppers})
    return fcast_df, acc


# ---------------------------------------------------------------------------
# Flat fallback
# ---------------------------------------------------------------------------

def _flat_fallback(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    steps: int,
    freq_minutes: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    last_val = float(train["y"].iloc[-1]) if not train.empty else 0.0
    last_ts = train["ds"].max()
    freq = pd.Timedelta(minutes=freq_minutes)
    future_ds = [last_ts + freq * (i + 1) for i in range(steps)]
    fcast_df = pd.DataFrame({
        "ds": future_ds,
        "yhat": last_val,
        "yhat_lower": last_val * 0.9,
        "yhat_upper": last_val * 1.1,
    })
    y_true = holdout["y"].values
    acc = _calc_accuracy(y_true, np.full(len(y_true), last_val))
    return fcast_df, acc


# ---------------------------------------------------------------------------
# Failure risk estimation
# ---------------------------------------------------------------------------

def _estimate_failure_risk(
    node_forecasts: dict[str, pd.DataFrame],
    freq_minutes: float,
) -> tuple[float, float | None]:
    """
    Estimate failure probability (0-1) and time-to-failure (hours).

    Logic
    -----
    - SNR forecast: breach = yhat < threshold
    - BER forecast: breach = yhat > threshold
    - Latency forecast: breach = yhat > threshold
    - For each metric, compute fraction of forecast window in breach
    - Combine with max(...) to get worst-case probability
    - TTF = first forecast step where any metric breaches
    """
    if not node_forecasts:
        return 0.0, None

    breach_fractions: list[float] = []
    first_breach_step: list[int] = []

    for metric, fcast_df in node_forecasts.items():
        if fcast_df is None or fcast_df.empty:
            continue
        y = fcast_df["yhat"].values
        thresh = THRESHOLDS.get(metric)
        if thresh is None:
            continue

        if metric == "snr":
            in_breach = y < thresh
        else:
            in_breach = y > thresh

        breach_fractions.append(float(in_breach.mean()))
        steps_to_breach = np.where(in_breach)[0]
        if len(steps_to_breach) > 0:
            first_breach_step.append(int(steps_to_breach[0]))

    if not breach_fractions:
        return 0.0, None

    # Logistic-like transform: fraction → probability
    max_fraction = max(breach_fractions)
    # Smooth mapping: 0 → 0, 0.5 → ~0.73, 1.0 → ~0.98
    prob = float(1 - np.exp(-3 * max_fraction))

    ttf = None
    if first_breach_step:
        earliest = min(first_breach_step)
        ttf = round(earliest * freq_minutes / 60, 2)

    return round(prob, 4), ttf


# ---------------------------------------------------------------------------
# Risk ranking table
# ---------------------------------------------------------------------------

def _build_risk_ranking(
    failure_prob: dict[str, float],
    time_to_failure: dict[str, float | None],
    df: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for node, prob in failure_prob.items():
        g = df[df["node_id"] == node]
        records.append({
            "node_id": node,
            "failure_probability": prob,
            "time_to_failure_hours": time_to_failure.get(node),
            "risk_level": _prob_to_risk(prob),
            "mean_snr": round(float(g["snr"].mean()), 2),
            "mean_ber": float(g["ber"].mean()),
            "mean_latency": round(float(g["latency"].mean()), 2),
        })
    rdf = pd.DataFrame(records)
    if not rdf.empty:
        rdf = rdf.sort_values("failure_probability", ascending=False).reset_index(drop=True)
    return rdf


def _prob_to_risk(p: float) -> str:
    if p >= 0.75:
        return "Critical"
    if p >= 0.50:
        return "High"
    if p >= 0.25:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lag_features(
    y: np.ndarray,
    lags: list[int],
    start: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a lagged-feature matrix from a 1-D array."""
    max_lag = max(lags)
    if start is None:
        start = max_lag
    X, target = [], []
    for i in range(start, len(y)):
        row = [y[i - l] for l in lags]
        X.append(row)
        target.append(y[i])
    return np.array(X), np.array(target)


def _calc_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"mae": float("nan"), "rmse": float("nan")}
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": round(mae, 6), "rmse": round(rmse, 6)}
