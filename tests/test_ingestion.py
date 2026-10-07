"""
tests/test_ingestion.py
-----------------------
Unit tests for the ingestion module.
"""

import sys
from pathlib import Path
import io
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion import ingest, generate_synthetic, REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_df():
    """Smallest valid DataFrame with all required columns."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="5min", tz="UTC"),
        "node_id": ["NODE_01"] * 20,
        "snr": np.linspace(28, 22, 20),
        "ber": np.linspace(1e-5, 5e-4, 20),
        "latency": np.linspace(40, 80, 20),
    })


@pytest.fixture
def synthetic_df():
    return generate_synthetic(n_nodes=3, hours=6, freq_minutes=5, seed=0)


# ---------------------------------------------------------------------------
# generate_synthetic
# ---------------------------------------------------------------------------

class TestGenerateSynthetic:
    def test_returns_dataframe(self):
        df = generate_synthetic(n_nodes=2, hours=4)
        assert isinstance(df, pd.DataFrame)

    def test_node_count(self):
        df = generate_synthetic(n_nodes=4, hours=6)
        assert df["node_id"].nunique() == 4

    def test_required_columns_present(self, synthetic_df):
        col_set = set(synthetic_df.columns)
        for col in REQUIRED_COLUMNS:
            assert col in col_set, f"Missing column: {col}"

    def test_row_count(self):
        hours, freq = 12, 10
        df = generate_synthetic(n_nodes=2, hours=hours, freq_minutes=freq)
        expected = int(hours * 60 / freq) * 2
        assert len(df) == expected

    def test_snr_range(self, synthetic_df):
        assert synthetic_df["snr"].between(-10, 60).all()

    def test_ber_range(self, synthetic_df):
        assert (synthetic_df["ber"] >= 0).all()
        assert (synthetic_df["ber"] <= 1).all()

    def test_latency_non_negative(self, synthetic_df):
        assert (synthetic_df["latency"] >= 0).all()

    def test_scenario_column_exists(self, synthetic_df):
        assert "scenario" in synthetic_df.columns

    def test_reproducible_with_seed(self):
        df1 = generate_synthetic(n_nodes=2, hours=4, seed=99)
        df2 = generate_synthetic(n_nodes=2, hours=4, seed=99)
        pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

class TestIngest:
    def test_basic_ingest(self, minimal_df):
        out = ingest(minimal_df)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 20

    def test_rolling_columns_created(self, minimal_df):
        out = ingest(minimal_df)
        assert "snr_roll_mean_5" in out.columns
        assert "ber_roll_mean_15" in out.columns
        assert "latency_ewma" in out.columns

    def test_trend_columns_created(self, minimal_df):
        out = ingest(minimal_df)
        assert "snr_trend" in out.columns
        assert "ber_trend" in out.columns
        assert "latency_trend" in out.columns

    def test_missing_required_column_raises(self, minimal_df):
        bad = minimal_df.drop(columns=["snr"])
        with pytest.raises(ValueError, match="Missing required columns"):
            ingest(bad)

    def test_column_alias_snr_db(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC"),
            "node_id": ["A"] * 10,
            "snr_db": [25.0] * 10,          # alias
            "ber": [1e-5] * 10,
            "latency": [40.0] * 10,
        })
        out = ingest(df)
        assert "snr" in out.columns

    def test_imputes_missing_values(self, minimal_df):
        minimal_df.loc[5:8, "snr"] = np.nan
        out = ingest(minimal_df)
        assert out["snr"].isna().sum() == 0

    def test_clips_snr_to_range(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC"),
            "node_id": ["A"] * 5,
            "snr": [100.0, -50.0, 25.0, 30.0, 28.0],
            "ber": [1e-5] * 5,
            "latency": [40.0] * 5,
        })
        out = ingest(df)
        assert out["snr"].max() <= 60
        assert out["snr"].min() >= -10

    def test_bad_timestamp_rows_dropped(self, minimal_df):
        minimal_df = minimal_df.copy()
        # Cast to object dtype first so pandas 2 accepts a string assignment
        minimal_df["timestamp"] = minimal_df["timestamp"].astype(object)
        minimal_df.loc[0, "timestamp"] = "not-a-date"
        out = ingest(minimal_df)
        assert len(out) == 19

    def test_csv_bytes_ingest(self, minimal_df):
        buf = io.BytesIO()
        minimal_df.to_csv(buf, index=False)
        csv_bytes = buf.getvalue()
        out = ingest(csv_bytes, fmt="csv")
        assert len(out) == 20

    def test_sorted_output(self, minimal_df):
        shuffled = minimal_df.sample(frac=1, random_state=0)
        out = ingest(shuffled)
        assert out["timestamp"].is_monotonic_increasing
