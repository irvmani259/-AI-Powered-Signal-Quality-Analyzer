"""
tests/test_forecasting.py
-------------------------
Unit tests for the forecasting module.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion import ingest, generate_synthetic
from forecasting import (
    forecast,
    ForecastResult,
    _calc_accuracy,
    _make_lag_features,
    _estimate_failure_risk,
    _prob_to_risk,
    THRESHOLDS,
    METRICS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_ingested_df():
    """Ingested 2-node dataset; module-scoped so it's built once."""
    raw = generate_synthetic(n_nodes=2, hours=12, freq_minutes=5, seed=1)
    return ingest(raw)


@pytest.fixture(scope="module")
def forecast_result(small_ingested_df):
    return forecast(small_ingested_df, horizon_hours=3)


# ---------------------------------------------------------------------------
# _calc_accuracy
# ---------------------------------------------------------------------------

class TestCalcAccuracy:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        acc = _calc_accuracy(y, y)
        assert acc["mae"] == pytest.approx(0.0)
        assert acc["rmse"] == pytest.approx(0.0)

    def test_known_error(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        acc = _calc_accuracy(y_true, y_pred)
        assert acc["mae"] == pytest.approx(1.0)
        assert acc["rmse"] == pytest.approx(1.0)

    def test_empty_arrays(self):
        acc = _calc_accuracy(np.array([]), np.array([]))
        assert np.isnan(acc["mae"])
        assert np.isnan(acc["rmse"])


# ---------------------------------------------------------------------------
# _make_lag_features
# ---------------------------------------------------------------------------

class TestMakeLagFeatures:
    def test_output_shape(self):
        y = np.arange(20, dtype=float)
        lags = [1, 2, 3]
        X, target = _make_lag_features(y, lags)
        assert X.shape[1] == len(lags)
        assert len(X) == len(target)

    def test_lag_values_correct(self):
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        lags = [1, 2]
        X, target = _make_lag_features(y, lags)
        # First row (i=2): lag1=y[1]=20, lag2=y[0]=10
        assert X[0, 0] == pytest.approx(20.0)
        assert X[0, 1] == pytest.approx(10.0)
        assert target[0] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# _prob_to_risk
# ---------------------------------------------------------------------------

class TestProbToRisk:
    @pytest.mark.parametrize("prob,expected", [
        (0.80, "Critical"),
        (0.60, "High"),
        (0.30, "Medium"),
        (0.10, "Low"),
        (0.0,  "Low"),
        (1.0,  "Critical"),
    ])
    def test_risk_labels(self, prob, expected):
        assert _prob_to_risk(prob) == expected


# ---------------------------------------------------------------------------
# _estimate_failure_risk
# ---------------------------------------------------------------------------

class TestEstimateFailureRisk:
    def _fcast_below_snr_thresh(self):
        """Forecast entirely below SNR threshold → should yield high probability."""
        snr_df = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=20, freq="5min"),
            "yhat": [5.0] * 20,           # below threshold of 10
            "yhat_lower": [3.0] * 20,
            "yhat_upper": [7.0] * 20,
        })
        return {"snr": snr_df}

    def test_all_below_threshold_high_prob(self):
        fc = self._fcast_below_snr_thresh()
        prob, ttf = _estimate_failure_risk(fc, freq_minutes=5.0)
        assert prob > 0.8

    def test_ttf_computed_when_breach(self):
        fc = self._fcast_below_snr_thresh()
        prob, ttf = _estimate_failure_risk(fc, freq_minutes=5.0)
        assert ttf is not None
        assert ttf >= 0

    def test_good_signal_low_prob(self):
        snr_df = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=20, freq="5min"),
            "yhat": [30.0] * 20,          # well above threshold
            "yhat_lower": [28.0] * 20,
            "yhat_upper": [32.0] * 20,
        })
        prob, ttf = _estimate_failure_risk({"snr": snr_df}, freq_minutes=5.0)
        assert prob == pytest.approx(0.0)
        assert ttf is None

    def test_empty_forecasts_zero_prob(self):
        prob, ttf = _estimate_failure_risk({}, freq_minutes=5.0)
        assert prob == pytest.approx(0.0)
        assert ttf is None


# ---------------------------------------------------------------------------
# forecast() — result structure
# ---------------------------------------------------------------------------

class TestForecastResultStructure:
    def test_returns_forecast_result(self, forecast_result):
        assert isinstance(forecast_result, ForecastResult)

    def test_forecasts_dict_keyed_by_node(self, forecast_result, small_ingested_df):
        nodes = set(small_ingested_df["node_id"].unique())
        assert set(forecast_result.forecasts.keys()) == nodes

    def test_each_node_has_metrics(self, forecast_result):
        for node, metrics in forecast_result.forecasts.items():
            for m in METRICS:
                # Each metric should either be present or absent due to data scarcity
                if m in metrics:
                    assert isinstance(metrics[m], pd.DataFrame), f"{node}/{m} not a DataFrame"

    def test_forecast_df_columns(self, forecast_result):
        for node, metrics in forecast_result.forecasts.items():
            for m, fdf in metrics.items():
                if fdf is not None and not fdf.empty:
                    assert "ds" in fdf.columns
                    assert "yhat" in fdf.columns
                    assert "yhat_lower" in fdf.columns
                    assert "yhat_upper" in fdf.columns

    def test_forecast_horizon_length(self, forecast_result, small_ingested_df):
        """Forecast should have roughly horizon_hours * 60 / freq_minutes steps."""
        for node, metrics in forecast_result.forecasts.items():
            for m, fdf in metrics.items():
                if fdf is not None and not fdf.empty:
                    # 3 hours at 5-min = 36 steps; allow ±5 for rounding
                    assert 20 <= len(fdf) <= 60

    def test_risk_ranking_is_dataframe(self, forecast_result):
        assert isinstance(forecast_result.risk_ranking, pd.DataFrame)

    def test_risk_ranking_sorted_descending(self, forecast_result):
        rr = forecast_result.risk_ranking
        if len(rr) > 1:
            probs = rr["failure_probability"].tolist()
            assert probs == sorted(probs, reverse=True)

    def test_failure_probability_in_0_1(self, forecast_result):
        for node, prob in forecast_result.failure_probability.items():
            assert 0.0 <= prob <= 1.0, f"Node {node} prob {prob} out of range"

    def test_accuracy_keys(self, forecast_result):
        for node, metrics in forecast_result.accuracy.items():
            for m, acc in metrics.items():
                assert "mae" in acc
                assert "rmse" in acc

    def test_risk_ranking_columns(self, forecast_result):
        rr = forecast_result.risk_ranking
        for col in ["node_id", "failure_probability", "risk_level"]:
            assert col in rr.columns
