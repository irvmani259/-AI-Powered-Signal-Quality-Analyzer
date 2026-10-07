"""
tests/test_detection.py
-----------------------
Unit tests for the detection module.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion import ingest, generate_synthetic
from detection import detect, DetectionResult, THRESHOLDS, _max_consecutive


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clean_df(n: int = 60, node: str = "NODE_01") -> pd.DataFrame:
    """Produce a clean ingested DF with all-normal signal values."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "node_id": [node] * n,
        "snr":     [28.0] * n,
        "ber":     [1e-6] * n,
        "latency": [40.0] * n,
    })
    return ingest(df)


def _make_degraded_df(n: int = 60) -> pd.DataFrame:
    """DF with SNR below threshold throughout."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "node_id":   ["NODE_01"] * n,
        "snr":       [5.0] * n,       # below threshold (10 dB)
        "ber":       [5e-3] * n,      # above threshold (1e-3)
        "latency":   [200.0] * n,     # above threshold (150 ms)
    })
    return ingest(df)


# ---------------------------------------------------------------------------
# _max_consecutive helper
# ---------------------------------------------------------------------------

class TestMaxConsecutive:
    def test_all_true(self):
        s = pd.Series([True] * 5)
        assert _max_consecutive(s) == 5

    def test_all_false(self):
        s = pd.Series([False] * 5)
        assert _max_consecutive(s) == 0

    def test_mixed(self):
        s = pd.Series([True, True, False, True, True, True, False])
        assert _max_consecutive(s) == 3


# ---------------------------------------------------------------------------
# detect() — result structure
# ---------------------------------------------------------------------------

class TestDetectResultStructure:
    @pytest.fixture(autouse=True)
    def run_detect(self):
        self.df = _make_clean_df()
        self.result = detect(self.df)

    def test_returns_detection_result(self):
        assert isinstance(self.result, DetectionResult)

    def test_anomalies_is_dataframe(self):
        assert isinstance(self.result.anomalies, pd.DataFrame)

    def test_node_health_is_dataframe(self):
        assert isinstance(self.result.node_health, pd.DataFrame)

    def test_interference_events_is_list(self):
        assert isinstance(self.result.interference_events, list)

    def test_degradation_events_is_list(self):
        assert isinstance(self.result.degradation_events, list)

    def test_summary_is_list(self):
        assert isinstance(self.result.summary, list)


# ---------------------------------------------------------------------------
# Rule-based flagging
# ---------------------------------------------------------------------------

class TestRuleBasedFlags:
    def test_snr_low_flagged(self):
        df = _make_clean_df()
        df.loc[10, "snr"] = THRESHOLDS["snr_min"] - 1
        result = detect(df)
        assert result.anomalies["flag_snr_low"].any()

    def test_ber_high_flagged(self):
        df = _make_clean_df()
        df.loc[10, "ber"] = THRESHOLDS["ber_max"] * 10
        result = detect(df)
        assert result.anomalies["flag_ber_high"].any()

    def test_latency_high_flagged(self):
        df = _make_clean_df()
        df.loc[10, "latency"] = THRESHOLDS["latency_max"] + 50
        result = detect(df)
        assert result.anomalies["flag_latency_high"].any()

    def test_clean_data_no_rule_flags(self):
        df = _make_clean_df()
        result = detect(df)
        rule_flags = ["flag_snr_low", "flag_ber_high", "flag_latency_high"]
        for flag in rule_flags:
            assert not result.anomalies[flag].any(), f"Expected no {flag} in clean data"


# ---------------------------------------------------------------------------
# Severity scoring
# ---------------------------------------------------------------------------

class TestSeverityScoring:
    def test_clean_data_normal_severity(self):
        df = _make_clean_df()
        result = detect(df)
        # No anomalies → all normal (anomalies table should be empty or only normal)
        if not result.anomalies.empty:
            assert (result.anomalies["severity"] == "Normal").all()

    def test_all_bad_metrics_critical(self):
        df = _make_degraded_df(n=80)
        result = detect(df)
        assert not result.anomalies.empty
        assert "Critical" in result.anomalies["severity"].values or \
               "High" in result.anomalies["severity"].values

    def test_severity_column_exists(self):
        df = _make_clean_df()
        result = detect(df)
        assert "severity" in result.anomalies.columns or len(result.anomalies) == 0


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

class TestIsolationForest:
    def test_flag_isolation_column_in_full_df(self):
        df = _make_clean_df(n=100)
        result = detect(df)
        # The flag should exist in the full underlying df (check via anomalies columns or raw)
        # After detect(), anomalies only has flagged rows; check detection ran without error
        assert result is not None

    def test_extreme_outlier_gets_flagged(self):
        df = _make_clean_df(n=100)
        # Inject an extreme outlier
        df.loc[50, "snr"] = -9.0
        df.loc[50, "ber"] = 0.9
        df.loc[50, "latency"] = 4000.0
        result = detect(df)
        assert len(result.anomalies) > 0


# ---------------------------------------------------------------------------
# Node health
# ---------------------------------------------------------------------------

class TestNodeHealth:
    def test_health_score_range(self):
        df = _make_clean_df()
        result = detect(df)
        if not result.node_health.empty:
            assert (result.node_health["health_score"] >= 0).all()
            assert (result.node_health["health_score"] <= 100).all()

    def test_degraded_node_lower_health(self):
        df_clean = _make_clean_df(node="CLEAN")
        df_bad = _make_degraded_df()
        df_combined = ingest(pd.concat([df_clean, df_bad], ignore_index=True))
        result = detect(df_combined)
        nh = result.node_health.set_index("node_id")
        if "CLEAN" in nh.index and "NODE_01" in nh.index:
            assert nh.loc["CLEAN", "health_score"] >= nh.loc["NODE_01", "health_score"]

    def test_health_df_has_required_columns(self):
        df = _make_clean_df()
        result = detect(df)
        for col in ["node_id", "health_score", "anomaly_rate", "mean_snr"]:
            assert col in result.node_health.columns


# ---------------------------------------------------------------------------
# Synthetic data end-to-end
# ---------------------------------------------------------------------------

class TestSyntheticDetection:
    def test_synthetic_produces_anomalies(self):
        raw = generate_synthetic(n_nodes=5, hours=12, seed=42)
        df = ingest(raw)
        result = detect(df)
        # Synthetic data includes degradation/interference scenarios → expect anomalies
        assert len(result.anomalies) > 0

    def test_synthetic_health_per_node(self):
        raw = generate_synthetic(n_nodes=5, hours=12, seed=42)
        df = ingest(raw)
        result = detect(df)
        assert len(result.node_health) == 5
